from Photo import Photo


class blur():
    def __init__(self, sblur, size):
        self.__sblur = sblur
        self.__size = size
        pass

class noise():
    def __init__(self, snoise, mean=0):
        self.__snoise = snoise
        self.__mean = mean
        pass


# Child class
class IP_master(Photo):
    def __init__(self, fp, fname, img, dtype):
        super().__init__(fp, fname, img)
        self.__dtype = dtype

        if self.__dtype == "matrix":
            # rewrite multiplication
            # k is a blur operator
            # X will be vectorized
            # blur(X) := K*x
            #
            pass 
        if self.__dtype == "signal":
            # rewrite multiplication
            # X * Y := X_f direct Y_f 
            # will add zeroes to allow for size
            #
            pass
        if self.__dtype == "complex":
            # rewrite multiplication 
            # Uses Transfer functions and complex numbers for signal representation
            #
            pass
