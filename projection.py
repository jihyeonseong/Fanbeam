import os
from torch.autograd import Function
from torch.utils.cpp_extension import load

module_path = os.path.dirname(__file__)
projection_op = load(
    "projection",
    sources=[
        os.path.join(module_path, "projection.cpp"),
        os.path.join(module_path, "projection_kernel.cu"),
    ],
)

class Projection(Function):
    @staticmethod
    def forward(ctx, img, nDetector, dDetector, dImage_x, dImage_y, nImage_x, nImage_y, nViews, DSD, DSO):
        return projection_op.projection(img, nDetector, dDetector, dImage_x, dImage_y, nImage_x, nImage_y, nViews, DSD, DSO)
    
def projection(img, nDetector, dDetector, dImage_x, dImage_y, nImage_x, nImage_y, nViews, DSD, DSO):
    return Projection.apply(img, nDetector, dDetector, dImage_x, dImage_y, nImage_x, nImage_y, nViews, DSD, DSO)