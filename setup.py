from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import setup

setup(
    name='flash_cuda',
    ext_modules=[
        CUDAExtension(
            name='flash_cuda',
            sources=['flash.cu'],
            extra_compile_args={
                'cxx': [],
                'nvcc': ['-O2']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
