

# Imports
from skimage import transform
from PIL.ExifTags import TAGS
import numpy as np
from typing import overload
from multiprocessing import SimpleQueue
from cv2 import cv2

from Photo import Photo

# Child class
class Image_Processing(Photo):
    def __init__(self, fp, fname, img):
        super().__init__(fp, fname, img)
        self.__blur = None
        self.__noise = None
        self.__kernel_size = (0, 0)
        self.__sigma_blur = 0
        self.__sigma_noise = 0
        self.__mean_noise = 0

    @property
    def blur(self):
        return self.__blur

    @property
    def noise(self):
        return self.__noise

    @property
    def kernel_size(self):
        return self.__kernel_size

    @property
    def sigma_blur(self):
        return self.__sigma_blur

    @property
    def mean_noise(self):
        return self.__mean_noise

    @property
    def sigma_noise(self):
        return self.__sigma_noise

    @kernel_size.setter
    def kernel_size(self, ksize):
        self.__kernel_size = ksize

    @sigma_blur.setter
    def sigma_blur(self, sigma_blur):
        self.__sigma_blur = sigma_blur

    @mean_noise.setter
    def mean_noise(self, mean_blur):
        self.__mean_blur = mean_blur

    @sigma_noise.setter
    def sigma_noise(self, sigma_noise):
        self.__sigma_noise = sigma_noise

    # Class Methods
    def blur_img(self, ksize, sigma_blur):
        self.kernel_size = ksize
        self.sigma_blur = sigma_blur
        blur = cv2.GaussianBlur(self.image, self.kernel_size, sigmaX=self.sigma_blur,
                                sigmaY=self.sigma_blur, borderType=cv2.BORDER_CONSTANT)
        self.__blur = Photo(fp=self.fp, fname="blured_"+self.fname, image=blur)

    def noise_img(self, mean_noise, sigma_noise):
        self.__mean_noise = mean_noise
        self.__sigma_noise = sigma_noise
        __n = np.random.normal(self.mean_noise, self.sigma_noise, self.shape)
        self.__noise = Photo(fp=self.fp, fname="noised_"+self.fname, image=__n)

    def alpha_blend_watermark(self, watermark_array, scale, location, alpha):
        def alpha_composite(img1, img2, alpha1):
            return img1*alpha1 + (1-alpha1)*img2

        def sp(xf): return [xf[:, :, ch] for ch in range(xf.shape[2])]

        h, w, c = watermark_array.shape
        h_hat, w_hat = int(scale*h), int(scale*w)
        water_test = transform.resize(
            watermark_array, (h_hat, w_hat, 4), cval=0)
        water_test = (water_test.astype(np.float32))

        blue, green, red = sp(self.image/255)
        blue1, green1, red1, a = sp(water_test/255)
        a = np.asarray(a, dtype='float32')

        x1, y1 = a.shape
        startx, starty = location
        x1, y1 = x1 + startx, y1 + starty

        rgb = self.image/255
        rgb[startx:x1, starty:y1, 2] = red1 * a + \
            (1.0 - a) * red[startx:x1, starty:y1]
        rgb[startx:x1, starty:y1, 1] = green1 * a + \
            (1.0 - a) * green[startx:x1, starty:y1]
        rgb[startx:x1, starty:y1, 0] = blue1*a + \
            (1.0 - a) * blue[startx:x1, starty:y1]

        rgb = (255*rgb).astype(np.uint8)

        return alpha_composite(self.image, rgb, alpha)
