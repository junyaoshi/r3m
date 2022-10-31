def scaling_factor_depth(cam_scale, hand_boxScale_o2n):
    scaling_factor = cam_scale / hand_boxScale_o2n
    depth = 1. / scaling_factor
    return depth