from ex1_utils import *
from gamma import gammaDisplay
import numpy as np
import matplotlib.pyplot as plt
import time


def histEqDemo(img_path: str, rep: int):
    img = imReadAndConvert(img_path, rep)
    imgeq, histOrg, histEq = hsitogramEqualize(img)

    # Display cumsum
    cumsum = np.cumsum(histOrg)
    cumsumEq = np.cumsum(histEq)
    plt.gray()
    plt.plot(range(256), cumsum, 'r')
    plt.plot(range(256), cumsumEq, 'g')
    plt.title("CumSum")

    # Display the images
    plt.figure()
    plt.title("Original Image")
    plt.imshow(img)

    plt.figure()
    plt.title("Histogram")
    plt.imshow(imgeq)
    plt.show()


def quantDemo(img_path: str, rep: int):
    img = imReadAndConvert(img_path, rep)
    st = time.time()

    img_lst, err_lst = quantizeImage(img, 3, 20)

    print("Time:%.2f" % (time.time() - st))
    print("Error 0:\t %f" % err_lst[0])
    print("Error last:\t %f" % err_lst[-1])

    plt.gray()
    plt.title("Quantized Image")
    plt.imshow(img_lst[0])
    plt.figure()
    plt.title("Quantized Image")
    plt.imshow(img_lst[-1])

    plt.figure()
    plt.title("Error Graph")
    plt.plot(err_lst, 'r')
    plt.show()


def main():
    print("ID:", myID())
    #img_path = 'beach.jpg'
    img_path = 'beach.jpg'

    # Basic read and display
    imDisplay(img_path, LOAD_GRAY_SCALE)
    imDisplay(img_path, LOAD_RGB)

    # Convert Color spaces
    img = imReadAndConvert(img_path, LOAD_RGB)
    yiq_img = transformRGB2YIQ(img)
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(img)
    ax[1].imshow(yiq_img)
    plt.title("RGB image converted into YIQ")
    plt.show()

    img1 = transformYIQ2RGB(yiq_img)
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(yiq_img)
    ax[1].imshow(img1)
    plt.title("YIQ image converted into RGB")
    plt.show()
    # Note the following: this is because we have negative values in YIQ however float's between [1,0] therefore rounds up to 0
    # Tried to play with it, didn't get better results
    # print(np.alltrue(img==img1))
    # print(np.allclose(img, img1))

    #exit(0)

    histEqDemo(img_path, LOAD_GRAY_SCALE)
    histEqDemo(img_path, LOAD_RGB)

    # Image Quantization
    quantDemo(img_path, LOAD_GRAY_SCALE)
    quantDemo(img_path, LOAD_RGB)


    # Gamma
    gammaDisplay(img_path, LOAD_GRAY_SCALE)



if __name__ == '__main__':
    main()
