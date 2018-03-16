#!python

from pathlib import Path

import torch
from torch.utils.ffi import create_extension

from _path import KALDI_ROOT


this_file = Path(__file__).parent

sources = ['src/latgen_lib.cc']
headers = ['src/latgen_lib.h']
defines = []

with_cuda = False

#if torch.cuda.is_available():
#    print('Including CUDA code.')
#    sources += ['src/my_lib_cuda.c']
#    headers += ['src/my_lib_cuda.h']
#    defines += [('WITH_CUDA', None)]
#    with_cuda = True

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

ffi = create_extension(
    '_ext.latgen_lib',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    language='c++',
    with_cuda=with_cuda,
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args
)

if __name__ == '__main__':
    import subprocess as sp

    print("## download & prepare model...")
    sp.run([f"scripts/mkgraph.sh {KALDI_ROOT}"], shell=True, check=True)

    print("\n## building latgen module...")
    ffi.build()
