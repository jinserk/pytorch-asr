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
    KALDI_ROOT + "/tools/openfst/include",
]
extra_compile_args = [
    "-std=c++11",
    "-w",
    "-fPIC",
]
libraries = list()
library_dirs = list()
runtime_library_dirs = list()
#extra_link_args = list()

kaldi_lib_root = KALDI_ROOT + "/src/lib"
library_dirs.append(kaldi_lib_root)
runtime_library_dirs.append(kaldi_lib_root)
libraries += [lib.name[3:-3] for lib in Path(kaldi_lib_root).rglob("libkaldi-*.so")]
#extra_link_args += [f"-l{lib}" for lib in libraries]
#extra_link_args.append(f"-Wl,-rpath={kaldi_lib_root}")

openfst_lib_root = KALDI_ROOT + "/tools/openfst/lib"
library_dirs.append(openfst_lib_root)
runtime_library_dirs.append(openfst_lib_root)
libraries.append("fst")
#extra_link_args.append("-lfst")
#extra_link_args.append(f"-Wl,-rpath={openfst_lib_root}")

print("\n## building latgen module...")
setup(
    name='torch_asr',
    ext_modules=[
        CppExtension(
            name='torch_asr._latgen_lib',
            sources=sources,
            libraries=libraries,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            runtime_library_dirs=runtime_library_dirs,
            extra_compile_args=extra_compile_args,
            #extra_link_args=extra_link_args
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

