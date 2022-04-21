


# Imports
from os import listdir
import os
from skimage import io
from PIL import Image
from PIL.ExifTags import TAGS

import matplotlib.pyplot as plt
import numpy as np
from typing import overload
import skimage as sk
import streamlit as st

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

    def add_to_book(self, book):
        book[self.__fname] = self

    def is_in_book(self, book):
        return self.__image in book.value

    def show(self):
        io.imshow((self.__image).astype(np.uint8))
        io.show()

    @staticmethod
    def load(fp, fname):
        img = io.imread(os.path.join(fp, fname))
        return img

    @staticmethod
    def save(fp, fname, image_array):
        io.imsave(os.path.join(fp, fname), image_array)

    @staticmethod
    def show_array(image_array, title=""):
        """
        modified for streamlit application
        io.title = title
        io.imshow(image_array.astype(np.uint8))
        io.show()
        """
        st.image(np.asarray(image_array, dtype=np.uint8), 
            channels="RGB", output_format="PNG", caption=title)

    @staticmethod
    def show_multiple(book):
        """
        Shows all Images within the dictionary object.
        """
        keys = list(book.keys())
        plt.subplot(2*(len(keys)//2), 2*(len(keys)//2), len(keys))
        i = 0
        for key in keys:
            plt.subplot(2*(len(keys)//2), len(keys)-(len(keys)//2), 1+i)
            io.imshow((book[key].image).astype(np.uint8))
            i += 1
        io.show()
