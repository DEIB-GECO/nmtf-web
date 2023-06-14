import numpy as np

def add_to_num(am, left):
    num = 0
    if left:
        num = np.linalg.multi_dot([am.association_matrix.transpose(), am.G_left, am.S])
    else:
        num = np.linalg.multi_dot([am.association_matrix, am.G_right, am.S.transpose()])
    return num


def add_to_den(am, G_left, left):
    den = 0
    if not G_left:
        if left:
            den += np.linalg.multi_dot([am.G_right, am.S.transpose(), am.G_left.transpose(), am.G_left, am.S])
        else:
            den += np.linalg.multi_dot([am.G_right, am.S, am.G_right.transpose(), am.G_right, am.S.transpose()])
    else:
        if left:
            den += np.linalg.multi_dot([am.G_left, am.S.transpose(), am.G_left.transpose(), am.G_left, am.S])
        else:
            den += np.linalg.multi_dot([am.G_left, am.S, am.G_right.transpose(), am.G_right, am.S.transpose()])
    return den