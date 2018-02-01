import os
from pathlib import Path
import numpy as np
import gzip
import bz2
import struct
import functools
import tempfile as tmp


def smart_open(filename, mode='rb', *args, **kwargs):
    '''
    Opens a file "smartly":
      * If the filename has a ".gz" or ".bz2" extension, compression is handled
        automatically;
      * If the file is to be read and does not exist, corresponding files with
        a ".gz" or ".bz2" extension will be attempted.
    '''
    readers = {'.gz': gzip.GzipFile, '.bz2': bz2.BZ2File}
    if 'r' in mode and not Path(filename).exists():
        for ext in readers:
            if Path(str(filename)+ext).exists():
                filename += ext
                break
    ext = Path(filename).suffix
    return readers.get(ext, open)(filename, mode, *args, **kwargs)


def read_string(f):
    s = ""
    while True:
        c = f.read(1).decode('utf-8')
        if c == "":
            raise ValueError("EOF encountered while reading a string.")
        if c == " ":
            return s
        s += c


def read_integer(f):
    n = ord(f.read(1))
    #return reduce(lambda x, y: x * 256 + ord(y), f.read(n)[::-1], 0)
    a = f.read(n)[::-1]
    try:
        return int.from_bytes(a, byteorder='big', signed=False)
    except:
        return functools.reduce(lambda x, y: x * 256 + ord(y), a, 0)
    #return functools.reduce(lambda x, y: x * 256 + ord(y), f.read(n)[::-1].decode('windows-1252'), 0)
    #try:
    #a=f.read(n)[::-1]
    #b=int.from_bytes(a, byteorder='big', signed=False)
    #print(a,type(a),b)
    #return functools.reduce(lambda x, y: x * 256 + ord(y), a[::-1], 0)
    #return functools.reduce(lambda x, y: x * 256 + ord(y), f.read(n)[::-1], 0)
    #except:
    #    return functools.reduce(lambda x, y: x * 256 + ord(y), f.read(n)[::-1].decode('windows-1252'), 0)


def read_vec_int(f):
    header = f.read(2).decode('utf-8')
    if header == "\0B":  # binary flag
        assert f.read(1).decode('utf-8') == '\4'  # int-size
        vec_size = np.frombuffer(f.read(4), dtype=np.int32, count=1)[0]  # vector dim
        # Elements from int32 vector are sored in tuples: (sizeof(int32), value),
        dt = [('size', np.int8), ('value', np.int32)]
        vec = np.frombuffer(f.read(int(vec_size) * 5), dtype=dt, count=vec_size)
        assert vec[0]['size'] == 4  # int32 size,
        ans = vec[:]['value']  # values are in 2nd column,
    else:  # ascii,
        arr = (header + f.readline().decode()).strip().split()
        try:
            arr.remove('[')
            arr.remove(']')  # optionally
        except ValueError:
            pass
        ans = np.array(arr, dtype=int)
    return ans


def read_matrix(f):
    header = f.read(2).decode('utf-8')
    if header != "\0B":
        raise ValueError("Binary mode header ('\0B') not found when attempting to read a matrix.")
    format = read_string(f)
    n_rows = read_integer(f)
    n_cols = read_integer(f)
    if format == "DM":
        data = struct.unpack("<%dd" % (n_rows * n_cols), f.read(n_rows * n_cols * 8))
        data = np.array(data, dtype="float64")
    elif format == "FM":
        data = struct.unpack("<%df" % (n_rows * n_cols), f.read(n_rows * n_cols * 4))
        data = np.array(data, dtype="float32")
    else:
        raise ValueError("Unknown matrix format '%s' encountered while reading; currently supported formats are DM (float64) and FM (float32)." % format)
    return data.reshape(n_rows, n_cols)


def read_matrix_shape(f):
    header = f.read(2).decode('utf-8')
    if header != "\0B":
        raise ValueError("Binary mode header ('\0B') not found when attempting to read a matrix.")
    format = read_string(f)
    n_rows = read_integer(f)
    n_cols = read_integer(f)
    if format == "DM":
        f.seek(n_rows * n_cols * 8, os.SEEK_CUR)
    elif format == "FM":
        f.seek(n_rows * n_cols * 4, os.SEEK_CUR)
    else:
        raise ValueError("Unknown matrix format '%s' encountered while reading; currently supported formats are DM (float64) and FM (float32)." % format)
    return n_rows, n_cols


def write_string(f, s):
    f.write((s + " ").encode('utf-8'))


def write_integer(f, a):
    s = struct.pack("<i", a)
    f.write(chr(len(s)).encode('utf-8') + s)


def write_matrix(f, data):
    f.write('\0B'.encode('utf-8'))      # Binary data header
    if str(data.dtype) == "float64":
        write_string(f, "DM")
        write_integer(f, data.shape[0])
        write_integer(f, data.shape[1])
        f.write(struct.pack("<%dd" % data.size, *data.ravel()))
    elif str(data.dtype) == "float32":
        write_string(f, "FM")
        write_integer(f, data.shape[0])
        write_integer(f, data.shape[1])
        f.write(struct.pack("<%df" % data.size, *data.ravel()))
    else:
        raise ValueError("Unsupported matrix format '%s' for writing; currently supported formats are float64 and float32." % str(data.dtype))


def read_ark(filename, limit=np.inf):
    """
    Reads the features in a Kaldi ark file.
    Returns a list of feature matrices and a list of the utterance IDs.
    """
    features = []
    uttids = []
    with smart_open(filename, "rb") as f:
        while True:
            try:
                uttid = read_string(f)
            except ValueError:
                break
            feature = read_matrix(f)
            features.append(feature)
            uttids.append(uttid)
            if len(features) == limit:
                break
    return features, uttids


def read_matrix_by_offset(arkfile, offset):
    with smart_open(arkfile, "rb") as g:
        g.seek(offset)
        feature = read_matrix(g)
    return feature


def read_scp(filename, limit=np.inf):
    """
    Reads the features in a Kaldi script file.
    Returns a list of feature matrices and a list of the utterance IDs.
    """
    features = []
    uttids = []
    with smart_open(filename, "r") as f:
        for line in f:
            uttid, pointer = line.strip().split(" ", 1)
            p = pointer.rfind(":")
            arkfile, offset = pointer[:p], int(pointer[p+1:])
            with smart_open(arkfile, "rb") as g:
                g.seek(offset)
                feature = read_matrix(g)
            features.append(feature)
            uttids.append(uttid)
            if len(features) == limit:
                break
    return features, uttids


def read_scp_info(filename, limit=np.inf):
    res = []
    with smart_open(filename, "r") as f:
        for line in f:
            uttid, pointer = line.strip().split()
            p = pointer.rfind(":")
            arkfile, offset = pointer[:p], int(pointer[p+1:])
            with smart_open(arkfile, "rb") as g:
                g.seek(offset)
                feat_len, feat_dim = read_matrix_shape(g)
            res.append((uttid, arkfile, offset, feat_len, feat_dim))
            if len(res) == limit:
                break
    return res


def read_scp_info_dic(filename, limit=np.inf):
    res = {}
    with smart_open(filename, "r") as f:
        for line in f:
            uttid, pointer = line.strip().split()
            p = pointer.rfind(":")
            arkfile, offset = pointer[:p], int(pointer[p+1:])
            with smart_open(arkfile, "rb") as g:
                g.seek(offset)
                feat_len, feat_dim = read_matrix_shape(g)
            res[uttid] = ((uttid, arkfile, offset, feat_len, feat_dim))
            if len(res) == limit:
                break
    return res


def write_ark(filename, features, uttids):
    """
    Takes a list of feature matrices and a list of utterance IDs,
      and writes them to a Kaldi ark file.
    Returns a list of strings in the format "filename:offset",
      which can be used to write a Kaldi script file.
    """
    pointers = []
#    with smart_open(filename, "wb") as f:
    with open(filename, "ab") as f:
        for feature, uttid in zip(features, uttids):
            write_string(f, uttid)
            pointers.append("%s:%d" % (filename, f.tell()))
            write_matrix(f, feature)
    return pointers


def write_scp(filename, uttids, pointers):
    """
    Takes a list of utterance IDs and a list of strings in the format "filename:offset",
      and writes them to a Kaldi script file.
    """
    with smart_open(filename, "w") as f:
        for uttid, pointer in zip(uttids, pointers):
            f.write("%s %s\n" % (uttid, pointer))


def tmp_write_ark(features, uttids):
    """
    Takes a list of feature matrices and a list of utterance IDs,
      and writes them to a Kaldi ark file.
    Returns a list of strings in the format "filename:offset",
      which can be used to write a Kaldi script file.
    """
    pointers = []
#    with smart_open(filename, "wb") as f:
    f = tmp.NamedTemporaryFile(mode="ab", suffix=".ark", delete=False)
    for feature, uttid in zip(features, uttids):
        write_string(f, uttid)
        pointers.append("%s:%d" % (f.name, f.tell()))
        write_matrix(f, feature)
    f.close()
    return f.name, pointers

