def get_position_list(rect, h_i, w_i):

    if not rect:
        return [0, 1, 0, 0]

    x, y, w, h = rect
    cx = x + w / 2

    if abs(cx - w_i/2) <= 5:
        return [1, 0, 0, 0]
    elif cx - w_i/2 < 0:
        return [0, 0, 1, 0]
    elif cx - w_i/2 > 0:
        return [0, 0, 0, 1]