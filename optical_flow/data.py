from PIL import Image
import numpy as np
import os


def read(file):
    TAG_FLOAT = 202021.25
    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == ".flo", "file ending is not .flo %r" % file[-4:]
    with open(file, "rb") as f:
        flo_number = np.fromfile(f, np.float32, count=1)[0]
        assert flo_number == TAG_FLOAT, (
            "Flow number %r incorrect. Invalid .flo file" % flo_number
        )
        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        data = np.fromfile(f, np.float32, count=2 * w[0] * h[0])
        flow = np.resize(data, (h[0], w[0], 2))

    return flow


if __name__ == "__main__":
    print(read("data/mpi_sintel/gt/frame_0001.flo").shape)
