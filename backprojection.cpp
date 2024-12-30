// #include <ATen/ATen.h>
#include <torch/extension.h>

torch::Tensor backprojection_op(const torch::Tensor &input, const float dDetector, 
                            const float dImage_x, const float dImage_y, const int nImage_x, const int nImage_y,
                            const float DSD, const float DSO);

#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

torch::Tensor backprojection(const torch::Tensor &input, const float dDetector, 
                            const float dImage_x, const float dImage_y, const int nImage_x, const int nImage_y,
                            const float DSD, const float DSO) {
  CHECK_INPUT(input);

  at::DeviceGuard guard(input.device());

  return backprojection_op(input, dDetector, dImage_x, dImage_y, nImage_x, nImage_y, DSD, DSO);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("backprojection", &backprojection, "backprojection (CUDA)");
}