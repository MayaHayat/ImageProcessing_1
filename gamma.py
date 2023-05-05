"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
# imports:
import cv2
from ex1_utils import LOAD_GRAY_SCALE, imReadAndConvert
title_window = 'Gamma correction'
from matplotlib import pyplot as plt
import numpy as np

trackbarName = "Gamma"

gammaSlideMax = 200 # since the slider is between 0-2
defaultValue = 100 # Corresponds to 1 which is OG image

# Define the function to apply gamma correction to the image.
def gammaCorrection(gamma):
    corrected = np.power(img / 255.0, gamma / 100.0) # Note all in float
    #print(img.dtype)
    corrected = np.uint8(corrected * 255)

    cv2.imshow("Image Gamma Correction", corrected)

def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """

    global img
    # Note to self: Make sure that the window name used in namedWindow() & createTrackbar() are the same
    # Load the image in BGR
    img = cv2.imread(img_path, rep)
    # Create the displaying window before creating the trackbar
    cv2.namedWindow("Image Gamma Correction")
    cv2.imshow("Image Gamma Correction", img)

    # Create the trackbar
    cv2.createTrackbar(trackbarName, "Image Gamma Correction", defaultValue, gammaSlideMax, gammaCorrection)
    # waitkey so the image doesn't close immediately, unless 'space' is pressed
    cv2.waitKey( )
    cv2.destroyAllWindows()


def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
