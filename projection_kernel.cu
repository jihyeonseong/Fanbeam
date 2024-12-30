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

#define PI 3.1415926535897f

struct ProjectionKernelParams {
    int nDetector;
    int nImage_x;
    int nImage_y;
    int nViews;
    float dDetector;
    float dImage_x;
    float dImage_y;
    float dRot;
    float dTheta;
    float DSD;
    float DSO;
};


template <typename scalar_t>
__global__ void projection_kernel(scalar_t *out, const scalar_t *input, 
                                  cudaTextureObject_t tex, 
                                  const ProjectionKernelParams p) {
    

    int detector_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int view_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (detector_idx >= p.nDetector || view_idx >= p.nViews)
        return;

    float dPixel_x = p.dImage_x / p.nImage_x;
    float dPixel_y = p.dImage_y / p.nImage_y;
    float dimgx0                = -0.5f * p.dImage_x + 0.5f * dPixel_x;
    float dimgy0                = -0.5f * p.dImage_y + 0.5f * dPixel_y;
    float dimgx1                = dimgx0 + p.dImage_x - dPixel_x;
    float dimgy1                = dimgy0 + p.dImage_y - dPixel_y;
    float dsamplingy0           = p.DSO - p.DSD;
    float dsamplingx0           = -(p.nDetector - 1) * p.dDetector * 0.5f;

    // Calculate angle for current view
    float angle = (view_idx * p.dTheta) * PI / 180.0f;
    
    // Calculate source position
    float source_x = p.DSO * cosf(angle);
    float source_y = p.DSO * sinf(angle);
    
    // Calculate detector position
    float d_beta = atanf(p.dDetector / p.DSD);
    float beta = (detector_idx - (p.nDetector ) / 2.0f) * d_beta;
    float detector_x = source_x + p.DSD * cosf(angle + beta + PI);
    float detector_y = source_y + p.DSD * sinf(angle + beta + PI);
    
    float direction_x = detector_x - source_x;
    float direction_y = detector_y - source_y;
    float direction_length = sqrtf(direction_x * direction_x + direction_y * direction_y);
    // Calculate ray direction
    float dir_x = direction_x / direction_length;
    float dir_y = direction_y / direction_length;

    // Ray sampling parameters
    const int nSampling = 1000;
    float step_size = p.DSD / nSampling;
    float sum = 0.0f;
    
    // Sample along the ray
    for (int i = 0; i < nSampling; i++) {
        float sample_x = source_x + i * step_size * dir_x;
        float sample_y = source_y + i * step_size * dir_y;
        
        // Convert to normalized texture coordinates
        if (sample_x >= dimgx0 && sample_x <= dimgx1 && sample_y >= dimgy0 && sample_y <= dimgy1) {
            float tex_x = sample_x / dPixel_x + p.nImage_x / 2;
            float tex_y = sample_y / dPixel_y + p.nImage_y / 2;
            sum += tex2DLayered<float>(tex, tex_x, tex_y, 0);
        }
    }
    // Store result in output array
    int out_idx =  p.nViews * detector_idx + view_idx;
    out[out_idx] = sum; //  / valid_samples;

}

torch::Tensor projection_op(const torch::Tensor &input, const int nDetector, const float dDetector, 
                            const float dImage_x, const float dImage_y, const int nImage_x, const int nImage_y, const int nViews,
                            const float DSD, const float DSO) {
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    ProjectionKernelParams p;

    auto x = input.contiguous();
    

    float dRot = 360.0f;
    p.nDetector = nDetector;
    p.dDetector = dDetector;
    p.dImage_x = dImage_x;
    p.dImage_y = dImage_y;
    p.nImage_x = nImage_x;
    p.nImage_y = nImage_y;
    p.nViews = nViews;
    p.dRot = dRot;
    p.dTheta = dRot / nViews;
    p.DSO = DSO;
    p.DSD = DSD;



    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    const cudaExtent extent = make_cudaExtent(nImage_x, nImage_y, 1);
    cudaArray_t array;
    cudaTextureObject_t texture = 0;

    cudaMalloc3DArray(&array, &channelDesc, extent, cudaArrayLayered);

    cudaMemcpy3DParms copyParams = {0};
    // copyParams.srcPtr = make_cudaPitchedPtr(x.data_ptr(), nImage_x * sizeof(float), nImage_y, nImage_y);
    copyParams.srcPtr = make_cudaPitchedPtr(x.data_ptr(), nImage_y * sizeof(float), nImage_y, nImage_x);
    copyParams.extent = make_cudaExtent(nImage_y, nImage_x, 1);

    copyParams.dstArray = array;
    // copyParams.extent = make_cudaExtent(nImage_x, nImage_y, 1);
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
    
    auto out = at::empty({ nDetector, nViews, }, x.options());

    dim3 block_size;
    dim3 grid_size;

    int threadsPerBlock = 4;
    block_size = dim3(threadsPerBlock, threadsPerBlock, 1);
    grid_size = dim3(ceil(float(nDetector) / threadsPerBlock), ceil(float(nViews) / threadsPerBlock), 1);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "projection_cuda", [&] {
        projection_kernel<scalar_t><<<grid_size, block_size, 0, stream>>>(out.data_ptr<scalar_t>(), 
                                                                            x.data_ptr<scalar_t>(), 
                                                                            texture, p);
    });

    return out;
}