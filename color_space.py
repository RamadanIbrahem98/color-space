from typing import Tuple
import numpy as np

np.seterr(invalid='ignore')

def RGB2XYZ(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts an RGB image to XYZ.

    Args:
        image: An RGB image.

    Returns:
        A tuple containing an X image, a Y image, and a Z image.
    """
    
    image = image / 255.0
    
    X = np.dot(image, [0.412453, 0.357580, 0.180423])
    Y = np.dot(image, [0.212671, 0.715160, 0.072169])
    Z = np.dot(image, [0.019334, 0.119193, 0.950227])

    return X, Y, Z

def XYZ2LUV(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    Converts an XYZ image to LUV.

    Args:
        X: An X image.
        Y: A Y image.
        Z: A Z image.

    Returns:
        An LUV image.
    """

    # These are Calculated using the same equations as u_dash and v_dash
    # with X, Y, Z being the reference values got from a lookup table
    # https://www.easyrgb.com/en/math.php
    # for observer Illumination being D65 and a CIE 2° Illuminant.
    U_REF = 0.19793943
    V_REF = 0.46831096

    L = np.where(Y > 0.008856, 116 * np.power(Y, 1 / 3) - 16, 903.3 * Y)

    u_dash = np.where((X + (15 * Y) + (3 * Z)) != 0, np.divide((4.0 * X), (X + (15.0 * Y) + (3.0 * Z))), 0)
    v_dash = np.where((X + (15 * Y) + (3 * Z)) != 0, np.divide((9.0 * Y), (X + (15.0 * Y) + (3.0 * Z))), 0)

    U = 13 * L * (u_dash - U_REF)
    V = 13 * L * (v_dash - V_REF)

    L = (255.0 / 100) * L
    U = (255.0 / 354) * (U + 134)
    V = (255.0 / 262) * (V + 140)

    LUV = np.dstack((L, U, V)).astype(np.uint8)

    return LUV

def RGB2LUV(image: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to LUV.

    Args:
        image: An RGB image.

    Returns:
        A LUV image.
    """
    X, Y, Z = RGB2XYZ(image)
    return XYZ2LUV(X, Y, Z)
