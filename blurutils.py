import cv2
import matplotlib.pyplot as plt
import numpy as np


def scatter_plot(red_pts, green_pts, X_name, Y_name, save_name):
    
    plt.scatter(red_pts[:, 0], red_pts[:, 1] , c='red')
    plt.scatter(green_pts[:, 0], green_pts[:, 1], c='green')
    plt.xlabel(X_name)
    plt.ylabel(Y_name)
    plt.legend(['Blur samples', 'NotBlur samples'])

    plt.savefig(f'{X_name}_{Y_name}_{save_name}.png', bbox_inches = 'tight', pad_inches = 0)
    
    return


def nth_percentile_laplacian(img, n=95):
    return np.percentile(cv2.convertScaleAbs(cv2.Laplacian(img,3, ksize=7)), n)


def canny(img):
    canny_img = cv2.Canny(img, 50, 100)
#     implt(canny_img)
    return canny_img


def canny_rowsum(canny_img):
    return np.mean(canny_img.sum(axis=0))


def canny_colsum(canny_img):
    return np.mean(canny_img.sum(axis=1))


def freq(img):
    fft = np.abs(np.fft.fft2(img))
    return np.abs(np.fft.fftshift(fft))


def mean_freq(freq):
#     M = np.max(np.abs(np.fft.fftshift(fft)))
#     return (len(fft[fft > M/1000]) / (fft.shape[0]*fft.shape[1]))*1000
#     return len(fft[fft > M/2000])

    return np.mean(freq)


def var_freq(freq):
#     M = np.max(np.abs(np.fft.fftshift(fft)))
#     return (len(fft[fft > M/1000]) / (fft.shape[0]*fft.shape[1]))*1000
#     return len(fft[fft > M/2000])

    return freq.var()


def laplacian(img, ddepth=3, ksize=7):
    # Find out the significance of these parameters
    # Not sure if abs is required, or np.abscaler is required.
    return np.abs(cv2.Laplacian(img, ddepth, ksize=ksize))


def var_laplacian(laplacian):
    return laplacian.var()


def max_laplacian(laplacian):
    return np.max(laplacian)


def mean_laplacian(laplacian):
    return np.mean(laplacian)
