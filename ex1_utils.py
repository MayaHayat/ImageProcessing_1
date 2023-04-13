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
from typing import List
from matplotlib import pyplot as plt
import numpy as np
LOAD_GRAY_SCALE = 1
LOAD_RGB = 2

# imports:
import cv2



# make sure to convert back to np.int ------------------------
def myID() -> int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 322515669


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """

    # cv2 reads image as BGR by default
    img = cv2.imread(filename)
    # plt.imshow(img)
    # plt.show()
    # exit(0)
    if img is None:
        raise ValueError("Couldn't open image")
    # if needed convert
    if representation == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif representation == 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("representation can be either 1 (Gray) or 2 (RGB)")

    # not sure how to do this
    final_img = cv2.normalize(img,  None, 0, 255, cv2.NORM_MINMAX)

    return img



def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """

    img = imReadAndConvert (filename, representation)
    plt.imshow(img)
    plt.show()

    pass


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """

    ogShape = imgRGB.shape
    conversion = np.array([[0.299, 0.587, 0.114],[0.59590059, -0.27455667, -0.32134392],[0.21153661, -0.52273617, 0.31119955]])
    # doing the dot product between the original image and the matrix but transposed and having it shaped as the original image was.
    final = np.dot(imgRGB.reshape(-1, 3), conversion.transpose()).reshape(ogShape)
    return final

    pass


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    ogShape = imgYIQ.shape
    # after saving the original shape of the image, we must save the conversion matrix given
    conversion = np.array([[0.299, 0.587, 0.114], [0.59590059, -0.27455667, -0.32134392], [0.21153661, -0.52273617, 0.31119955]])
    # Next we use the np.linalg.inv function to find [conversion]^-1
    inverseMatrix = np.linalg.inv(conversion)
    # doing the dot product between the original image and the inversed matrix but transposed and having it shaped as the original image was.
    final = np.dot(imgYIQ.reshape(-1, 3), inverseMatrix.transpose()).reshape(ogShape)
    return final

    pass


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """


    pass


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """


    pass
