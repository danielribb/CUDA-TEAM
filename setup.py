from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='flash_attention',
    ext_modules=[
        CUDAExtension('flash_attention', [
            'build.cpp',      
            'flashfwd2v1.cu',  
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
