import numpy as np
from data import get_flow_data
import matplotlib.pyplot as plt
import flow_vis


def lucas_kanade_block(Ix, Iy, It):
    return np.zeros((Ix.shape[0], Ix.shape[1], 2))


def lucas_kanade(init_img, target_img, window_size=(5, 5)):
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
            Ix_ij = Ix[i : window_size[0], j : window_size[1]]
            Iy_ij = Iy[i : window_size[0], j : window_size[1]]
            It_ij = It[i : window_size[0], j : window_size[1]]
            result[i : window_size[0], j : window_size[1]] = lucas_kanade_block(
                Ix_ij, Iy_ij, It_ij
            )

    return result, (slice_x, slice_y)


def estimate(algorithm=lucas_kanade):
    result_flows = []
    prev_image, _ = get_flow_data(1)
    for i in range(2, 51):
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
