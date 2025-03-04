from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CppExtension

setup(
    name='custom_attention',
    ext_modules=[
        CUDAExtension(
            name='custom_attention',
            sources=[
                'attention.cpp',         # Arquivo de ligação C++
                'attention_cuda.cu',     # Arquivo CUDA
            ],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
