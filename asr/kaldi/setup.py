#!python

from pathlib import Path
import subprocess as sp
from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CppExtension

from _path import KALDI_ROOT


# making graph
print("## download & prepare model...")
sp.run([f"scripts/mkgraph.sh {KALDI_ROOT}"], shell=True, check=True)

# building latget_lib
sources = ['src/latgen_lib.cc']

include_dirs = [
    KALDI_ROOT + "/src",
    KALDI_ROOT + "/tools/openfst/src/include",
]
extra_compile_args = [
    "-std=c99",
    "-std=c++11",
    "-w",
    "-fPIC",
]
library_dirs = list()
extra_link_args = list()

kaldi_lib_root = KALDI_ROOT + "/src"
for lib in Path(kaldi_lib_root).rglob("libkaldi-*.so"):
    library_dirs.append(str(lib.parent))
    extra_link_args.append(f"-l{str(lib.name)[3:-3]}")
    extra_link_args.append(f"-Wl,-rpath={str(lib.parent)}")

openfst_lib_root = KALDI_ROOT + "/tools/openfst/lib"
library_dirs.append(openfst_lib_root)
extra_link_args.append("-lfst")
extra_link_args.append(f"-Wl,-rpath={openfst_lib_root}")

print("\n## building latgen module...")
setup(
    name='torch_asr',
    ext_modules=[
        CppExtension(
            name='torch_asr._latgen_lib',
            sources=sources,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

