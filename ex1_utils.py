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
from sklearn.utils import shuffle
# imports:
import cv2
from typing import List
from matplotlib import pyplot as plt
import numpy as np

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2





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
    if img is None:
        raise ValueError("Couldn't open image")
    # if needed convert
    if representation == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif representation == 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("representation can be either 1 (Gray) or 2 (RGB)")

    img = img.astype(np.float64)
    # not sure how to do this
    img_norm = cv2.normalize(img,  None, 0, 1, cv2.NORM_MINMAX)
    return img_norm



def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """

    img = imReadAndConvert(filename, representation)
    #plt.imshow(img)
    if representation == 1:
        plt.imshow(img, cmap='gray')
    elif representation == 2:
        plt.imshow(img)

    plt.title("Displayed image with representation={0}".format(representation))
    plt.show()



def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """

    conversion = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321],[0.212, -0.523, 0.311]])
    final = np.dot(imgRGB, conversion.transpose())
    #norm_img = cv2.normalize(final, None, 0, 1, cv2.NORM_MINMAX)
    # norm_img = (final - np.min(final)) / (np.max(final) - np.min(final))
    #print(np.max(final), np.min(final))

    return final


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    conversion = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321],[0.212, -0.523, 0.311]])
    # Next we use the np.linalg.inv function to find [conversion]^-1
    inverse_mat = np.linalg.inv(conversion)
    final = np.dot(imgYIQ, inverse_mat.transpose())
    # #norm_img = (final - np.min(final)) / (np.max(final) - np.min(final))
    #print(np.max(final), np.min(final))

    return final


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    img = np.copy(imgOrig)
    #print(np.max(img))

    # Note: The YIQ representation is sometimes employed in color image processing transformations. For example, applying a histogram equalization directly to the channels in an RGB image would alter the color balance of the image. Instead, the histogram equalization is applied to the Y channel of the YIQ or YUV representation of the image, which only normalizes the brightness levels of the image. - Wikipedia
    colour = len(img.shape)
    if colour == 3:
        YIQ_transform = transformRGB2YIQ(imgOrig)
        img = YIQ_transform[:, :, 0]

    # img range is [0,1], need to convert it to [0,255]
    # print(np.min(imgOrig) , np.max(imgOrig) , imgOrig.dtype)
    norm_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    # has to be converted as it is now between 0 and 255 -> less memory needed
    norm_img = norm_img.astype(np.uint8)
    hist, bins = np.histogram(norm_img.flatten(), bins=256, range=(0, 256), density=False)
    #plt.bar(bins[:-1], hist, color='pink', width=1)
    #plt.show()

    cumSum = np.cumsum(hist)
    cumSumNorm = cumSum/np.max(cumSum)
    LUT = (cumSumNorm * 255).astype(np.uint8)

    norm_img_flatten = norm_img.flatten()
    # eq_img_list = [T[f] for f in img_list]
    imEq = LUT[norm_img_flatten]
    imEq = np.reshape(imEq, norm_img.shape)
    histEq, binsEq = np.histogram(imEq.flatten(), bins=256, range=(0, 256), density=False)
    imgEq = imEq.astype(np.float64)
    imEq = imgEq / 255

    if colour == 3:
        YIQ_transform[:, :, 0] = imEq
        imEq = transformYIQ2RGB(YIQ_transform)

    # plt.imshow(imgOrig)
    # plt.show()
    # plt.imshow(imEq)
    # plt.show()
    return imEq, hist, histEq,

# this function was taken from www.tutorialspoint.com
def mse(img1, img2):
   h, w = img1.shape
   diff = cv2.subtract(img1, img2)
   err = np.sum(diff**2)
   mse = err/(float(h*w))
   return mse


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):

    # check if nums are within range
    if nQuant > 256:
        raise ValueError("nQuant value must be 256 or less")

    if nIter < 1:
        raise ValueError("nIter value must be 1 or more")

    # Convert if needed
    img = np.copy(imOrig)
    if len(imOrig.shape) == 3:
        YIQ_transform = transformRGB2YIQ(imOrig)
        img = YIQ_transform[:, :, 0]

    qImage_i ,mse_list = [], []
    # Initialize z by using the following function, could as well use np.arange, takes section size instead of num of sections
    z = np.linspace(0, 1, nQuant+1)

    for i in range(nIter):
        q = [0]*nQuant
        resetIm = np.zeros_like(img)
        curBoundary = [0] * (nQuant + 1)
        curBoundary[nQuant] = 1
        for j in range(nQuant):
            # Take all nums within the section (creates a truth table
            q[j] = np.mean(img[(img >= z[j]) & (img <= z[j+1])])
            if j > 0:
                curBoundary[j] = (q[j-1] + q[j]) / 2
                #print(z, curBoundary, " z")

        if np.alltrue(z == curBoundary):
            break
        z = curBoundary

        for j in range(nQuant):
            resetIm[(img >= z[j]) & (img <= z[j+1])] = q[j]
            #print(q, " q")

        mse_list.append(mse(img, resetIm))

        if len(imOrig.shape) == 3:
            cpy_Img = YIQ_transform.copy()
            cpy_Img[:, :, 0] = resetIm
            resetIm = transformYIQ2RGB(cpy_Img)
        qImage_i.append(resetIm)

    return qImage_i, mse_list


