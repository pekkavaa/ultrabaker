"""
Based on https://github.com/colour-science/colour/blob/develop/colour/models/rgb/ycocg.py
"""

import numpy as np

def vecmul(m, v) -> np.ndarray:
    return np.matmul(m, v[..., None]).squeeze(-1)

MATRIX_RGB_TO_YCOCG: np.ndarray = np.array(
    [
        [1 / 4, 1 / 2, 1 / 4],
        [1 / 2, 0, -1 / 2],
        [-1 / 4, 1 / 2, -1 / 4],
    ]
)

MATRIX_YCOCG_TO_RGB: np.ndarray = np.array(
    [
        [1, 1, -1],
        [1, 0, 1],
        [1, -1, -1],
    ]
)


def RGB_to_YCoCg(RGB) -> np.ndarray:
    return vecmul(MATRIX_RGB_TO_YCOCG, RGB)


def YCoCg_to_RGB(YCoCg) -> np.ndarray:
    return vecmul(MATRIX_YCOCG_TO_RGB, YCoCg)

if __name__ == "__main__":
    print(RGB_to_YCoCg(np.array([1.0, 1.0, 1.0])))
    # array([ 1.,  0.,  0.])
    print(RGB_to_YCoCg(np.array([0.75, 0.5, 0.5])))
    # array([ 0.5625,  0.125 , -0.0625])
    print(YCoCg_to_RGB(np.array([1.0, 0.0, 0.0])))
    # array([ 1.,  1.,  1.])
    print(YCoCg_to_RGB(np.array([0.5625, 0.125, -0.0625])))
    # array([ 0.75,  0.5 ,  0.5 ])