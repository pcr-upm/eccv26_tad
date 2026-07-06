from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import torch.utils.cpp_extension

# Hack to bypass CUDA version check
torch.utils.cpp_extension._check_cuda_version = lambda *args, **kwargs: None

# Define the extension
sources = [
    "opentad/models/bricks/2d_sparse_v3_bind.cpp",
    "opentad/models/bricks/2d_sparse_v3.cu",
    "opentad/models/bricks/implicit_gemm_v2.cu",
]

# Ensure sources exist
for source in sources:
    if not os.path.exists(source):
        raise FileNotFoundError(f"Source file not found: {source}")

setup(
    name="opentad",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="opentad_sparse_ops",
            sources=sources,
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
