import argparse
import math
import cv2
import numpy as np
import skimage.io
import scipy

def median_filter(image, rad):
    height, width = image.shape
    new_shape = (height, width)
    res = np.zeros(new_shape, dtype=float)
    for y in range(height):
        for x in range(width):
            sum = 0
            for i in range(-rad, rad + 1):
                for j in range(-rad, rad + 1):
                    x_ = max(0, min(x+i, width-1))
                    y_ = max(0, min(y+j, height-1))
                    sum += image[y_,x_]
            res[y, x] = sum / ((2*rad  + 1) * (2*rad  + 1))

    return res


def gaussian_filter(image, rad, sigma_d):
    height, width = image.shape
    new_shape = (height, width)
    res = np.zeros(new_shape, dtype=float)
    kernel_size = 2*rad + 1
    kernel = np.zeros((kernel_size, kernel_size))
    for i in range(-rad, rad + 1):
        for j in range(-rad, rad + 1):
            kernel[rad+i, rad+j] = np.exp(-(i**2 + j**2) / (2 * sigma_d**2))
    kernel /= (2 * math.pi * sigma_d**2)
    for y in range(height):
        for x in range(width):
            res_val = 0
            for i in range(-rad, rad + 1):
                for j in range(-rad, rad + 1):
                    x_ = max(0, min(x+i, width-1))
                    y_ = max(0, min(y+j, height-1))
                    res_val += kernel[rad+i, rad+j] * image[y_, x_]
            res[y, x] = res_val

    return res


def bilateral_filter(image, sigma_d, sigma_r):
    sigma_d = float(sigma_d)
    sigma_r = float(sigma_r)
    rad = math.ceil(3 * sigma_d)
    height, width = image.shape
    new_shape = (height, width)
    res = np.zeros(new_shape, dtype=float)
    add_img = np.zeros((height + 2*rad, width + 2*rad))
    for i in range(height + 2 * rad):
        for j in range(width + 2 * rad):
            add_img[i, j] = image[max(0, min(i-rad, height-1)), max(0, min(j-rad, width-1))]
    kernel = np.zeros((2*rad + 1, 2*rad + 1))
    for i in range(2*rad + 1):
        for j in range(2*rad + 1):
            kernel[i, j] = np.exp(-((i-rad)**2 + (j-rad)**2) / (2 * sigma_d**2))
    radius = kernel.shape[0] // 2
    for i in range(height):
        for j in range(width):
            tmp = -np.square(add_img[i: i + 2*radius + 1, j: j + 2*radius + 1] - np.ones((2*radius + 1, 2*radius + 1)) * image[i, j])
            curr_filter = kernel * np.exp(tmp / (2 * sigma_r**2))
            sum = curr_filter.sum()
            res[i, j] = (add_img[i: i + 2 * radius + 1, j: j + 2 * radius + 1] * curr_filter).sum() / sum
    res = res.astype(int)
    
    return res


def mse(image_1, image_2):
    image1 = image_1.astype(np.float64)
    image2 = image_2.astype(np.float64)
    res = np.sum(np.square((np.subtract(image1, image2))))
    res /= image_1.shape[0] * image_1.shape[1]
    res = res * 255 * 255
    return res 


def psnr(image_1, image_2):
    _mse = mse(image_1, image_2)
    if (_mse == 0):
        return None
    res = 10 * np.log10(255 * 255 / _mse)
    return res


def ssim(image_1, image_2):
    L = 1
    c1 = (0.01*L) ** 2
    c2 = (0.03*L) ** 2
    image1 = image_1.astype(np.float64)
    image2 = image_2.astype(np.float64)
    mean_1 = np.mean(image1)
    mean_2 = np.mean(image2)
    sigma_1 = np.var(image1)
    sigma_2 = np.var(image2)
    sigma12 = np.cov(image1.flatten(), image2.flatten(), bias=True)[0, 1]
    tmp_1 = (2 * mean_1 * mean_2 + c1) * (2 * sigma12 + c2)
    tmp_2 = (mean_1 ** 2 + mean_2 ** 2 + c1) * (sigma_1 + sigma_2 + c2)
    res = tmp_1 / tmp_2

    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Main_task_2',
        description='task_2_ImPr',
        epilog='Text at the bottom of help',
    )
    parser.add_argument('command', help='Command description')
    parser.add_argument('parameters', nargs='*')
    args = parser.parse_args()

    
    if args.command == 'mse':
        image_1 = skimage.io.imread(args.parameters[0])
        image_2 = skimage.io.imread(args.parameters[1])
        image1 = image_1 / 255
        image2 = image_2 / 255
        if len(image1.shape) == 3:
            image1 = image1[:, :, 0]
        if len(image2.shape) == 3:
            image2 = image2[:, :, 0]
        res = mse(image1, image2)
        print(res)
        
    elif args.command == 'psnr':
        image_1 = skimage.io.imread(args.parameters[0])
        image_2 = skimage.io.imread(args.parameters[1])
        image1 = image_1 / 255
        image2 = image_2 / 255
        if len(image1.shape) == 3:
            image1 = image1[:, :, 0]
        if len(image2.shape) == 3:
            image2 = image2[:, :, 0]
        res = psnr(image1, image2)
        print(res)
        
    elif args.command == 'ssim':
        image_1 = skimage.io.imread(args.parameters[0])
        image_2 = skimage.io.imread(args.parameters[1])
        image1 = image_1 / 255
        image2 = image_2 / 255
        if len(image1.shape) == 3:
            image1 = image1[:, :, 0]
        if len(image2.shape) == 3:
            image2 = image2[:, :, 0]
        res = ssim(image1, image2)
        print(res)

    elif args.command == 'median':
        image_ = skimage.io.imread(args.parameters[1])
        image = image_ / 255
        if len(image.shape) == 3:
            image = image[:, :, 0]
        rad = int(args.parameters[0])
        res = median_filter(image, rad)
        res = np.clip(res, 0, 1)
        res = (res * 255).astype(np.uint8)
        skimage.io.imsave(args.parameters[2], res)

    elif args.command == 'gauss':
        image_ = skimage.io.imread(args.parameters[1])
        image = image_ / 255
        if len(image.shape) == 3:
            img1 = image[:, :, 0]
        sigma = float(args.parameters[0])
        rad = math.ceil(3 * sigma)
        res = gaussian_filter(img1, rad, sigma)
        res = np.clip(res, 0, 1)
        res = (res * 255).astype(np.uint8)
        skimage.io.imsave(args.parameters[2], res)

    elif args.command == 'bilateral':
        image_ = skimage.io.imread(args.parameters[2])
        if (image_.shape[2] == 3):
            image = (image_[:, :, 0])
        res = bilateral_filter(image, args.parameters[0], args.parameters[1])
        cv2.imwrite(args.parameters[3], res)