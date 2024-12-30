#include <torch/types.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_texture_types.h>

#include <math.h>

#include <iostream>
#include <cuda_fp16.h>
#include <cufft.h>

#include <cmath>
#include <cstring>


#define PI 3.1415926535897f

struct BackprojectionKernelParams {
    int nDetector;
    int nImage_x;
    int nImage_y;
    int nViews;

    float dDetector;
    float dImage_x;
    float dImage_y;
    float dRot;
    float dTheta;
    float DSD;  // Add Distance Source to Detector
    float DSO;  // Add Distance Source to Object center
};

torch::Tensor filter_sinogram_cuda(const torch::Tensor &input, const char* filter_type,
                                 const float dDetector, const float DSD) {
    int nDetector = input.size(0);
    int nViews = input.size(1);
    // Create frequency array
    auto freqs = torch::fft_fftfreq(nDetector).to(input.device()).to(input.dtype());
    freqs = freqs.reshape({-1, 1});
    // Calculate sampling parameters
    float sampling_interval = dDetector / DSD;  // Angular sampling interval
    // Create ramp filter |ω|
    auto filt = torch::abs(freqs);
    // Apply proper fan-beam weighting
    auto detector_pos = torch::linspace(-dDetector * (nDetector-1)/2.0f,
                                      dDetector * (nDetector-1)/2.0f,
                                      nDetector, torch::device(input.device()));
    auto gamma = torch::atan2(detector_pos, torch::full_like(detector_pos, DSD));
    auto cos_gamma = torch::cos(gamma).reshape({-1, 1});
    // Fan-beam filter includes:
    // 1. Ramp filter |ω|
    // 2. cos(γ) weighting
    // 3. DSD scaling
    filt = filt * cos_gamma * DSD;
    // Optional: Apply smoothing window for noise reduction
    if (strcmp(filter_type, "hamming") == 0) {
        auto window = 0.54f + 0.46f * torch::cos(PI * freqs / 0.5f);
        filt = filt * window;
    }
    // Apply filtering
    auto output = torch::empty_like(input);
    for (int i = 0; i < nViews; i++) {
        auto view = input.select(1, i);
        auto view_fft = torch::fft_fft(view);
        auto filtered_view = torch::real(torch::fft_ifft(view_fft * filt.squeeze()));
        output.select(1, i) = filtered_view;
    }
    return output;
}


template <typename scalar_t>
__global__ void backprojection_kernel(scalar_t *out, const scalar_t *input, 
                                  int k, cudaTextureObject_t tex, 
                                  const BackprojectionKernelParams p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= p.nImage_x || j >= p.nImage_y)
        return;

    // Calculate pixel position in world coordinates
    float dPixel_x = p.dImage_x / p.nImage_x;
    float dPixel_y = p.dImage_y / p.nImage_y;
    float dimgx0 = -0.5f * p.dImage_x + 0.5f * dPixel_x;
    float dimgy0 = -0.5f * p.dImage_y + 0.5f * dPixel_y;
    
    float pixel_x = dimgx0 + i * dPixel_x;
    float pixel_y = dimgy0 + j * dPixel_y;

    // Calculate angle for current view
    float angle = (k * p.dTheta) * PI / 180.0f;
    
    // Calculate source position
    float source_x = p.DSO * cosf(angle);
    float source_y = p.DSO * sinf(angle);
    
    // Calculate vector from source to pixel
    float dx = pixel_x - source_x;
    float dy = pixel_y - source_y;

    // Calculate beta (angle between central ray and pixel ray)
    float pixel_angle = atan2f(dy, dx);
    float beta = pixel_angle - angle + PI;

    while (beta > PI) beta -= 2.0f * PI;
    while (beta < -PI) beta += 2.0f * PI;

    float d_beta = atanf(p.dDetector / p.DSD);  // angular spacing between detectors
    float center_index = (p.nDetector - 1) / 2.0f;
    float detector_idx = beta / d_beta + center_index;
    
    detector_idx = roundf(detector_idx);
    if (detector_idx < 0 || detector_idx >= p.nDetector) {
        return;
    }

    if (detector_idx >= 0 && detector_idx < p.nDetector) {
        // Calculate weight based on distance
        float U = sqrtf(dx * dx + dy * dy);  // distance from source to pixel
        float V = sqrtf(powf(pixel_x - (source_x - p.DSD * cosf(angle + beta)), 2) +
                       powf(pixel_y - (source_y - p.DSD * sinf(angle + beta)), 2)); 

        float weight = 1.0f / (U + V+ 1e-7f);
        
        // Sample from sinogram texture
        float value = tex2DLayered<float>(tex, k + 0.5f, detector_idx + 0.5f, 0);
        
        // Accumulate weighted value
        atomicAdd(&out[j * p.nImage_x + i], value * weight * (p.dTheta * PI / 180.0f) * 0.5f);
        // printf("value: %f, weight: %f, p.dTheta: %, PI: %f, out: %f\n", value, weight, p.dTheta, PI, out[j * p.nImage_x + i]);
    }
}




torch::Tensor backprojection_op(const torch::Tensor &input, const float dDetector, 
                            const float dImage_x, const float dImage_y, const int nImage_x, const int nImage_y,
                            const float DSD, const float DSO) {
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    BackprojectionKernelParams p;

    auto x = input.contiguous();
    x = filter_sinogram_cuda(x, "hamming", dDetector, DSD);
    
    float dRot = 360.0f;
    p.nDetector = x.size(0);
    p.dDetector = dDetector;
    p.dImage_x = dImage_x;
    p.dImage_y = dImage_y;
    p.nImage_x = nImage_x;
    p.nImage_y = nImage_y;
    p.nViews = x.size(1);
    p.dRot = dRot;
    p.dTheta = dRot / p.nViews;
    p.DSD = DSD;
    p.DSO = DSO;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaExtent extent = make_cudaExtent(p.nViews, p.nDetector, 1);
    cudaArray_t array;
    cudaTextureObject_t texture = 0;

    cudaMalloc3DArray(&array, &channelDesc, extent, cudaArrayLayered);

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr(x.data_ptr(), p.nViews * sizeof(float), p.nViews, p.nDetector);
    copyParams.dstArray = array;
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyDeviceToDevice;
    cudaMemcpy3D(&copyParams);

    // Create resource descriptor
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = array;


    // Specify texture object parameters
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    // Create texture object
    cudaCreateTextureObject(&texture, &resDesc, &texDesc, NULL);

    auto out = at::empty({nImage_y, nImage_x}, x.options());
    out.zero_();

    dim3 block_size;
    dim3 grid_size;

    int threadsPerBlock = 2;
    block_size = dim3(threadsPerBlock, threadsPerBlock, 1);
    grid_size = dim3(ceil(float(nImage_x)/threadsPerBlock), ceil(float(nImage_y)/threadsPerBlock), 1);
    for (int k = 0; k < p.nViews; k++) {
    // for (int k = 89; k < 90; k++) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "backprojection_cuda", [&] {
            backprojection_kernel<scalar_t><<<grid_size, block_size, 0, stream>>>(out.data_ptr<scalar_t>(), 
                                                                                  x.data_ptr<scalar_t>(), 
                                                                                  k, texture, p);
        });
    }
    return out;
}
