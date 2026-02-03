import numpy as np

def hillas(charge_coords):
    """
    Calculates the Hillas parameters for a camera image.
    :param charge_coords: np.ndarray, (3, 1600) three lists; x, y, charge
    :return: list of Hillas parameters for event
    :rtype: List
    """
    x_data = charge_coords[0]
    y_data = charge_coords[1]
    charges_data = charge_coords[2]
    x = 0
    y = 0
    x2 = 0
    y2 = 0
    xy = 0
    CHARGE = 0
    CHARGE = np.nansum(charges_data)
    x = np.nansum(x_data * charges_data)
    y = np.nansum(y_data * charges_data)
    x2 = np.nansum(x_data ** 2 * charges_data)
    y2 = np.nansum(y_data ** 2 * charges_data)
    xy = np.nansum(x_data * y_data * charges_data)
    x /= CHARGE
    y /= CHARGE
    x2 /= CHARGE
    y2 /= CHARGE
    xy /= CHARGE
    S2_x = x2 - x ** 2
    S2_y = y2 - y ** 2
    S_xy = xy - x * y
    d = S2_y - S2_x
    a = (d + np.sqrt(d ** 2 + 4 * S_xy ** 2)) / (2 * S_xy)
    b = y - a * x
    z = np.sqrt(d ** 2 + 4 * (S_xy ** 2))
    width = np.sqrt((S2_y + a ** 2 * S2_x - 2 * a * S_xy) / (1 + a ** 2))
    length = np.sqrt((S2_x + a ** 2 * S2_y + 2 * a * S_xy) / (1 + a ** 2))
    miss = b / np.sqrt(1 + a ** 2)
    dis = np.sqrt(x ** 2 + y ** 2)
    psi = np.arctan(((d + z) * x + 2 * S_xy * x) / ((2 * S_xy - (d - z)) * x))
    q_coord = (x - x_data) * (x / dis) + (y - y_data) * (y / dis)
    q = np.nansum(q_coord * charges_data) / CHARGE
    q2 = np.nansum(q_coord ** 2 * charges_data) / CHARGE
    azwidth = q2 - q ** 2
    alpha = np.arcsin(miss / dis)
    return [a, b, x, y, width, length, miss, dis, azwidth, alpha, psi]