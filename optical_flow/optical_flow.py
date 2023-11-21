import numpy as np
from data import get_flow_data
import matplotlib.pyplot as plt
import flow_vis
from tqdm import tqdm


def lucas_kanade_block(Ix, Iy, It, eps=1e-6):
    A = np.stack([Ix.reshape(-1), Iy.reshape(-1)]).T
    uv = (np.linalg.pinv(A.T @ A + eps) @ A.T @ It.reshape(-1, 1)).reshape(2)
    result = np.tile(uv, (Ix.shape[0], Ix.shape[1], 1))
    return result


def lucas_kanade(init_img, target_img, window_size=(30, 30)):
    slice_x, slice_y = [
        init_img.shape[i] - init_img.shape[i] % window_size[i] for i in range(2)
    ]
    init_img = init_img[:slice_x, :slice_y, :]
    target_img = target_img[:slice_x, :slice_y, :]

    # compute gradients
    Ix, Iy, _ = np.gradient(init_img)
    Ix = np.mean(Ix, axis=-1)  # dump the color dimension
    Iy = np.mean(Iy, axis=-1)
    It = (target_img - init_img).mean(axis=-1)

    result = np.zeros((init_img.shape[0], init_img.shape[1], 2))

    for i in range(init_img.shape[0] // window_size[0]):
        for j in range(init_img.shape[1] // window_size[1]):
            from_i, to_i = i * window_size[0], window_size[0] * (i + 1)
            from_j, to_j = j * window_size[1], window_size[1] * (j + 1)
            Ix_ij = Ix[from_i:to_i, from_j:to_j]
            Iy_ij = Iy[from_i:to_i, from_j:to_j]
            It_ij = It[from_i:to_i, from_j:to_j]
            result[from_i:to_i, from_j:to_j] = lucas_kanade_block(Ix_ij, Iy_ij, It_ij)
    return result, (slice_x, slice_y)


def estimate(algorithm=lucas_kanade):
    result_flows = []
    prev_image, _ = get_flow_data(1)
    for i in tqdm(range(2, 51)):
        image, _ = get_flow_data(i)
        result = algorithm(prev_image, image)
        prev_image = image
        result_flows.append(result[0])
    return result_flows


if __name__ == "__main__":
    flows = estimate()
    for flow in flows:
        flow_pic = flow_vis.flow_to_color(flow)
        plt.imshow(flow_pic)
        plt.show()
        plt.close()
