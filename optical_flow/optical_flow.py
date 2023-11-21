import numpy as np
from data import get_flow_data


def lucas_kanade(init_img, target_img):
    result = np.zeros((init_img.shape[0], init_img.shape[1], 2))
    return result


def estimate(algorithm=lucas_kanade):
    result_flows = []
    prev_image, _ = get_flow_data(1)
    for i in range(2, 51):
        image, _ = get_flow_data(i)
        result = algorithm(prev_image, image)
        prev_image = image
        result_flows.append(result)
    return result_flows


if __name__ == "__main__":
    print(estimate())
