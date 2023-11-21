from PIL import Image
import numpy as np
import os
import flow_vis
from matplotlib import pyplot as plt


def read_gt(file):
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


def get_flow_data(i):
    png_path = f"data/mpi_sintel/imgs/frame_{str(i).zfill(4)}.png"
    image = Image.open(png_path)
    image = np.array(image.getdata()) / 255.0

    gt_path = f"data/mpi_sintel/gt/frame_{str(i-1).zfill(4)}.flo"
    try:
        gt = read_gt(gt_path)
    except AssertionError:
        print(f"exception at {i}")
        gt = np.zeros((image.shape[0], image.shape[1], 2))

    return image, gt


if __name__ == "__main__":
    for i in range(1, 49):
        _, gt = get_flow_data(i)
        flow = flow_vis.flow_to_color(gt)
        plt.imshow(flow)
        plt.show()
        plt.close()
