import sys
from pathlib import Path, PurePath
from cffi import FFI
from ctypes import *
from subprocess import check_output
import numpy

cur_dir = Path(__file__).parents[0]

# get Kaldi libraries path
script = cur_dir.joinpath("path.sh")
kaldi_root = check_output([f"source {script} && echo $KALDI_ROOT"], shell=True).strip().decode()
kaldi_src = PurePath(kaldi_root, 'src')
kaldi_lib = [x for x in Path(kaldi_src).rglob("*.so")]
for lib in kaldi_lib:
    cdll.LoadLibrary(lib)

ffi = FFI()
ffi.set_unicode(enabled_flag=True)

# C functions declaration
ffi.cdef("""\
void initialize(float beam, int max_active, int min_active, float acoustic_scale,
                int allow_partial, char *fst_in_filename, char *words_in_filename);
void decode(float *loglikes, int *sizes, char **texts, int **words, int **alignments);\
""")

# open liblatgen.so
liblatgen = cur_dir.joinpath("liblatgen.so")
default_graph = cur_dir.joinpath("graph", "TLG.fst")
default_words = cur_dir.joinpath("graph", "words.txt")

_C = ffi.dlopen(str(liblatgen))

## create the dictionary mapping ctypes to numpy dtypes.
ctype2dtype = {}

for prefix in ('int', 'uint'):
    for log_bytes in range(4):
        m, n = 8*(2**log_bytes), 2**log_bytes
        ctype2dtype[f"{prefix}{m}_t"] = numpy.dtype(f"{prefix[0]}{n}")

ctype2dtype['int'] = numpy.dtype('i4')
ctype2dtype['float'] = numpy.dtype('f4')
ctype2dtype['double'] = numpy.dtype('f8')
# print( ctype2dtype )


def asarray(ffi, ptr, length):
    ## get the canonical C type of the elements of ptr as a string.
    T = ffi.getctype(ffi.typeof( ptr ).item)
    if T not in ctype2dtype:
        raise RuntimeError( "Cannot create an array for element type: %s" % T )
    return numpy.frombuffer( ffi.buffer( ptr, length*ffi.sizeof( T ) ), ctype2dtype[T] )

#def test():
#    from cffi import FFI
#    ffi = FFI()
#
#    N = 10
#    ptr = ffi.new( "float[]", N )
#
#    arr = asarray( ffi, ptr, N )
#    arr[:] = numpy.arange( N )
#
#    for i in range( N ):
#        print( arr[i], ptr[i] )

def initialize(beam=16.0, max_active=8000, min_active=200, acoustic_scale=1.0, allow_partial=True,
               fst_file=str(default_graph), wd_file=str(default_words)):
    fst_in_filename = fst_file.encode('ascii')
    wd_in_filename = wd_file.encode('ascii')
    _C.initialize(beam, max_active, min_active, acoustic_scale, allow_partial,
                  fst_in_filename, wd_in_filename)

def decode(loglikes):
    c_sizes = ffi.new("int[]", loglikes.shape)
    c_loglikes = ffi.cast("float *", ffi.from_buffer(loglikes))

    # N: batch size, RxC: R frames for C classes
    N, R, C = loglikes.shape
    c_texts = [ffi.new("char[]", 2048) for n in range(N)]
    c_text_list = ffi.new("char *[]", c_texts)
    c_words = [ffi.new("int[]", [-1]*R) for n in range(N)]
    c_words_list = ffi.new("int *[]", c_words)
    c_alignments = [ffi.new("int[]", [-1]*R) for n in range(N)]
    c_alignment_list = ffi.new("int *[]", c_alignments)

    _C.decode(c_loglikes, c_sizes, c_text_list, c_words_list, c_alignment_list)

    # currently each utterance in batch returns the best lattice only
    texts = [[ffi.string(c_text_list[n]).decode()] for n in range(N)]
    words = [[asarray(ffi, c_words_list[n], R)] for n in range(N)]
    alignment = [[asarray(ffi, c_alignment_list[n], R)] for n in range(N)]

    return texts, words, alignment


#if __name__ == "__main__":
#    ffibuilder.compile(verbose=True)
