import numpy as np


def projection_point(p, f):
    m = [
        [f, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, 1, 0]
    ]
    p_h = [p, 1]
    projection = np.array(m) @ np.array(p_h)
    p_img = [
        projection[0]/projection[-1],
        projection[1]/projection[-1]
        ]
    return p_img

if __name__ == '__main__':
    p = [200, 100, 50]
    f = 50
    print(projection_point(p, f))
