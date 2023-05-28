import numpy as np

def create_anchor(ratios=[0.5, 1, 2], base_sizes=[8, 16, 32]):
    anchor_base = np.zeros((len(ratios) * len(base_sizes), 4), dtype=np.float32)
    for i in range(len(ratios)):
        for j in range(len(base_sizes)):
            h = 16 * base_sizes[j] * np.sqrt(ratios[i])
            w = 16 * base_sizes[j] * np.sqrt(1. / ratios[i])

            index = i * len(base_sizes) + j
            anchor_base[index, 0] = - h / 2.
            anchor_base[index, 1] = - w / 2.
            anchor_base[index, 2] = h / 2.
            anchor_base[index, 3] = w / 2.
    return anchor_base

def create_all_anchor(anchor_base, feat_stride, height, width):

    shift_x             = np.arange(0, width * feat_stride, feat_stride)
    shift_y             = np.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y    = np.meshgrid(shift_x, shift_y)
    shift               = np.stack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel(),), axis=1)
    A       = anchor_base.shape[0]
    K       = shift.shape[0]
    anchor  = anchor_base.reshape((1, A, 4)) + shift.reshape((K, 1, 4))
    anchor  = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor
