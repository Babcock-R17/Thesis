


# Imports
from os import listdir
import os
from cv2 import CV_32FC3
from skimage import io
from PIL import Image
from PIL.ExifTags import TAGS

import matplotlib.pyplot as plt
import numpy as np
from typing import overload
import skimage as sk
import streamlit as st
import cv2 

# Parent Class
class Photo():
    def __init__(self, fp, fname, image):
        self.__fp = fp
        self.__fname = fname
        self.__image = image.astype(np.float32)
        self.__shape = np.asarray(image).shape
        self.__title = fname

    @property
    def fp(self):
        return self.__fp

    @property
    def fname(self):
        return self.__fname

    @property
    def image(self):
        return self.__image

    @property
    def shape(self):
        return self.__shape

    @property
    def width(self):
        return self.__shape[0]

    @property
    def height(self):
        return self.__shape[1]

    @property
    def channel(self):
        try:
            return self.__shape[2]
        except ValueError as e:
            print(e)
            return None

    @property
    def title(self):
        return self.__title

    @fp.setter
    def fp(self, fp):
        self.__fp = fp

    @fname.setter
    def fname(self, fname):
        self.__fname = fname

    @image.setter
    def image(self, image):
        self.__image = image

    @shape.setter
    def shape(self, shape):
        self.__shape = shape

    @title.setter
    def title(self, title):
        self.__title = title

    # Class Methods

    def describe(self):
        name = self.fname
        shape = self.shape
        print("Image {} has a shape of {} ".format(name, shape))

    def extraxt_metadata(**kwargs):
        # open the image
        image = Image.open(kwargs["from_file"])
        # extracting the exif metadata
        res_string = ""
        exifdata = image.getexif()
        # looping through all the tags present in exifdata
        for tagid in exifdata:
            # getting the tag name instead of tag id
            tagname = TAGS.get(tagid, tagid)
            # passing the tagid to get its respective value
            value = exifdata.get(tagid)
            # printing the final result
            res_string += f"{tagname:25}: {value}\t"
        return res_string

    def show(self, title="", size=10 ,figsize=(5, 5), **kwargs ):
        fig = plt.figure(figsize=figsize)
        plt.imshow( self.image.astype(np.uint8) )
        plt.title(title, size=size)
        plt.axis("off")
        if 'save' in kwargs.keys():
            plt.savefig(kwargs['save'])
        plt.show()

    @staticmethod
    def load(fp, fname):
        img = io.imread(os.path.join(fp, fname))
        return img

    @staticmethod
    def save(fp, fname, image_array):
        io.imsave(os.path.join(fp, fname), image_array)

    @staticmethod
    def show_array(image, title="", size=10 ,figsize=(5, 5), **kwargs ):
        fig = plt.figure(figsize=figsize)
        plt.imshow(  image.astype(np.uint8) )
        plt.title(title, size=size)
        plt.axis("off")
        if 'save' in kwargs.keys():
            plt.savefig(kwargs['save'])
        plt.show()

    @staticmethod
    def transform(X, **kwargs):
        keys, values = kwargs.keys() , kwargs.values()
        if "rgb2gray" in keys and kwargs["rgb2gray"]:
            X = np.float32(X)
            if len(X.shape) == 3:
                X = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)

        if "crop" in keys:
            # select Region of Interest
            roi = kwargs['crop']
            roi = roi.strip()
            roi = roi.replace(" ", "")
            xrange, yrange = roi.split(",")
            xl, xu = xrange.split(':')
            yl, yu = yrange.split(':')
            xl, xu = int(xl), int(xu)
            yl, yu = int(yl), int(yu)
            
            print("selected ROI: [{}:{} by {}:{}]".format( xl, xu, yl, yu) )
            if len(X.shape) == 3:
                X = X[xl:xu, yl:yu, :]
            else:
                X = X[xl:xu, yl:yu]


        return X

