import os
from torch.autograd import Function
from torch.utils.cpp_extension import load

module_path = os.path.dirname(__file__)
backprojection_op = load(
    "backprojection",
    sources=[
        os.path.join(module_path, "backprojection.cpp"),
        os.path.join(module_path, "backprojection_kernel.cu"),
    ],
)

class BackProjection(Function):
    @staticmethod
    def forward(ctx, y, dDetector, dImage_x, dImage_y, nImage_x, nImage_y, DSD, DSO):
        return backprojection_op.backprojection(y, dDetector, dImage_x, dImage_y, nImage_x, nImage_y, DSD, DSO)    

def backprojection(y, dDetector, dImage_x, dImage_y, nImage_x, nImage_y, DSD, DSO):
    return BackProjection.apply(y, dDetector, dImage_x, dImage_y, nImage_x, nImage_y, DSD, DSO)    