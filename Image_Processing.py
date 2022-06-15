
__author__ = 'Jasen Babcock'

# Imports
import matplotlib
from matplotlib.animation import PillowWriter
from skimage import transform
from PIL.ExifTags import TAGS
import numpy as np
from typing import overload
import scipy as sp
from cv2 import cv2, mean
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import pandas as pd
import seaborn as sns
from Photo import Photo

from scipy.sparse.linalg import svds
from smithnormalform import matrix, snfproblem, z
import math
import matplotlib.animation as animation
from scipy.optimize import least_squares
import scipy.stats as st


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
        self.__blur_matrix = 0
        self.__psf = None

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
    @property
    def blur_matrix(self):
        return self.__blur_matrix
    
    @property
    def psf(self):
        return self.__psf

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

    @blur_matrix.setter
    def blur_matrix(self, blur_matrix):
        self.__blur_matrix = blur_matrix
    
    @psf.setter
    def psf(self, psf):
        self.__psf = psf

    # Class Methods
    def blur_img(self, ksize, sigma_blur):
        #setting properties
        p_size = int(ksize[0] ) # assuming ksize is symetric and odd
        image_shape = self.image.shape

        # Dont need channel information yet since im assuming each color has the same blur
        image_shape = image_shape[0:2] 
        self.kernel_size = ksize
        self.sigma_blur = sigma_blur        
        psf = self.psf( sigma_L= np.array([sigma_blur, sigma_blur ]), p_size=p_size )
        #k = self.blur_matrix( psf=psf, image_shape=image_shape) 
        #self.blur_matrix = k.astype( dtype=np.float64 )
        self.psf = psf

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
    
    def fftn(self):
        X_f = sp.fft.fftn(self.image/255)
        return X_f

    def ifftn( X_f):
        X = sp.fft.ifftn( X_f)
        return X


    @staticmethod
    def display(X, title="", size=10, figsize=(5,5), fname="save_fig",  color_map='winter',show=True,  **type):
        def check_and_plot(keys, type, X ):
            def animate(fig, fname=fname):
                # rotate the axes and update
                writer = animation.PillowWriter(fps=15)
                with writer.saving(fig, outfile=fname+".gif", dpi=50):
                    # Iterate over frames
                    elevation = 60
                    sigma = 30
                    for angle in range(0, 360):
                        h = angle%180 
                        if angle < 180:
                            elevation = 60*np.exp(-(h*h)/(2*sigma**2)) + 30
                        else: 
                            elevation = -60*np.exp(-(h*h)/(2*sigma**2)) + 90
                            
                        ax.view_init(elevation, angle)
                        writer.grab_frame()
                writer.finish()
            
            
            
            
            
            m, n = X.shape
            if ("manifold" in keys) and type["manifold"]:
                fig = plt.figure(figsize=figsize)
                ax = plt.axes(projection ='3d')
                ax.set_xlabel("Pixel Location over i")
                ax.set_ylabel("Pixel Location over j")
                ax.set_zlabel("Pixel Value (0, 255)")
                u, v = np.mgrid[0:m:1, 0:n:1]
                x = u
                y = v
                z = X[u,v]
                surf = ax.plot_surface(x, y, z, cmap=color_map)
                # add a color bar
                plt.axis("off")
                fig.colorbar(surf )
                plt.title(title)
                if "animate" in keys and type["animate"]:
                    animate(fig)



            if ("signals" in keys) and type["signals"]:
                # create a mesh grid
                capt = '''
                The output, analogously to fft, contains the term for zero frequency
                in the low-order corner of all axes, the positive frequency terms in the first half of all axes, the term for the Nyquist frequency in the middle of all axes
                and the negative frequency terms in the second half of all axes, in order of decreasingly negative frequency.
                '''
                fig = plt.figure(figsize=figsize)
                ax = plt.axes(projection ='3d')
                ax.set_xlabel("Freq")
                ax.set_ylabel("Freq")
                ax.set_zlabel("Amplitude")
                ax.set_zorder(255)
                X_f = np.fft.fft2(X)

                x, y = np.mgrid[0:m:1, 0:n:1]
                X_fshift = np.fft.fftshift(X_f[x, y])
                
                z = np.log(np.abs(X_fshift)**2)
                surf = ax.plot_surface(x, y, z, cmap=color_map)

                # add a color bar
                plt.axis("off")
                fig.colorbar(surf)
                
                plt.title(title+"\nSignal Transform")
                if "animate" in keys and type["animate"]:
                    animate(fig)



            if ("cluster_amp" in keys) and type["cluster_amp"]:
                X_f = np.fft.fft2(X)
                x, y = np.mgrid[0:m:1, 0:n:1]
                X_fshift = np.fft.fftshift(X_f[x, y])
                df = pd.DataFrame( np.log(np.abs(X_fshift)**2) )

                Z_link = sp.cluster.hierarchy.linkage(df, method='centroid',  metric='euclidean')
                Z_df = pd.DataFrame(Z_link  )

                #print(Z_df.describe() )
                cophenet = sp.cluster.hierarchy.cophenet(Z_link ) 

                dist_ij = pd.DataFrame(sp.spatial.distance.squareform(cophenet) )

                #print(dist_ij.head)
                #print(dist_ij.describe())


                sns.set_theme(color_codes=True)
                sns.clustermap(df, cmap=color_map )
                plt.title(title+"\nClustered Amplitudes")
                if "animate" in keys and type["animate"]:
                    animate(fig)

            if ("cluster_freq" in keys) and type["cluster_freq"]:
                X_f = np.fft.fft2(X)
                x, y = np.mgrid[0:m:1, 0:n:1]
                X_fshift = np.fft.fftshift(X_f[x, y])
                angles = np.angle(X_fshift)
                df = pd.DataFrame(angles)

                Z_link = sp.cluster.hierarchy.linkage(df, method='centroid',  metric='euclidean')
                Z_df = pd.DataFrame(Z_link  )

                print(Z_df.describe() )
                cophenet = sp.cluster.hierarchy.cophenet(Z_link ) 

                dist_ij = pd.DataFrame(sp.spatial.distance.squareform(cophenet) )

                print(dist_ij.head)
                print(dist_ij.describe())


                sns.set_theme(color_codes=True)
                sns.clustermap(df, cmap=color_map)
                plt.title(title+"\nClustered Frequencies")
                if "animate" in keys and type["animate"]:
                    animate(fig)            

            
            if ("spec" in keys) and type['spec']:
                # create a mesh grid
                capt = '''
                The output, analogously to fft, contains the term for zero frequency
                in the low-order corner of all axes, the positive frequency terms in the first half of all axes, the term for the Nyquist frequency in the middle of all axes
                and the negative frequency terms in the second half of all axes, in order of decreasingly negative frequency.
                '''
                fig = plt.figure(figsize=figsize)
                ax = plt.axes()
                X_f = np.fft.fft2(X)
                # color version scheme winter
                im1 = ax.imshow(np.log(np.abs(np.fft.fftshift(X_f))**2) , cmap=color_map )
                cb1 = fig.colorbar(im1, location='bottom'  )
                cb1.set_label(' log( | fft_shifted(x_f) | ) **2 ', rotation=0)
                
                plt.axis("off")
                plt.title(title+"\nPower Spectrogram")
                if "animate" in keys and type["animate"]:
                    animate(fig)

        keys = type.keys()
        if len(X.shape) == 3:
            """
            Use sp function to split color channels and hstack results
            """
            X = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)
        
        if ("style" in keys):
            with plt.style.context(style=type['style'], after_reset=True):
                print("\nSelected Style is \t "+ type["style"])
                check_and_plot(keys, type, X)
        else:
            check_and_plot(keys, type, X)

        if "save" in keys:
            print("Saving figure to .... ",type["save"])
            plt.savefig(type["save"])

        if show:
            plt.show()
        else:
            print(f"{fname} Loaded...")

        pass
    @staticmethod
    def blur_matrix(psf, image_shape):
        """
        Input: psf is a matrix that describes the point spread function, psf must be odd for now
                image_shape is a tuple describing the shape of the blurring matrix. 
        
        """
        if psf.shape[0] > min(image_shape[0:2]):
            print("psf must be smaller than or equal to the image dimensions")
        c = np.array(psf.shape)//2 # Define the Center of my PSF
        p_0, p_1 = psf.shape
        n, m = image_shape[0:2]
        k_matrix = np.zeros( (m*n, m*n ))
        for j in range(m):
            j_lb, j_ub = max(c[1]- j, 0), min(m - j + c[1], p_1)
            for i in range(n):
                i_lb, i_ub = max(c[0] - i, 0), min(n - i + c[0], p_0)
                template = np.zeros(image_shape[0:2]) # clear template
                template[ max(i - c[0], 0): 1 + min(n, i + c[0] ), 
                        max(j - c[1], 0): 1 + min(m,   j + c[1])] = psf[i_lb:i_ub, j_lb:j_ub ]
                k_matrix[:, i%n + m*j ] = template.reshape( template.size  )
        return k_matrix.astype(dtype=np.float64)

    @staticmethod
    def psf( sigma_L=np.array([0, 0]), p_size=3):
        center = np.array( [ p_size//2 , p_size//2 ]  )
        psf = np.zeros((p_size, p_size))

        for j in range(psf.shape[1]):
            for i in range(psf.shape[0]):
                loc = np.array([i, j])
                x = loc - center
                diag_m =  x * x * 1/(sigma_L*sigma_L)
                psf[i, j] = np.exp(-1*np.sum(diag_m)/2)
        psf = psf / np.sum(psf)
        return psf


    @staticmethod
    def threshold_filter(y, I, percentile=95):
            """
            Input: accepts a recieved RGB image with a watermark that may contain blur + noise, 
                    the original clean RGB image with no watermark.
                    Transforms RGB image with watermark and RGB image clean into frequency domain. 
            returns: A signal that is highly uncorellated to the original image. 
            """
            y_f = sp.fft.fftn(y/255)
            I_f = sp.fft.fftn(I/255)
            # Create a filter
            m, n = y_f.shape
            amplitude = np.abs(y_f).reshape(m*n)
            ir = np.abs(y_f*I_f).reshape(m*n)
            threshold = np.percentile(ir, q=percentile)
            for i in range(len(amplitude)):
                if ir[i] < threshold:
                    y_f.reshape(m*n)[i] = 0
                else:
                    pass
            return y_f.reshape(m,n) # returns the fft of the correlated signals with CI of ___ 

    @staticmethod
    def distortion(A, B):
        Ag = Photo.transform(A, rgb2gray=True)
        Bg = Photo.transform(B, rgb2gray=True)
        return 100*np.linalg.norm(Ag-Bg, 'fro')/np.linalg.norm(Bg, 'fro')

    @staticmethod
    def decomp(X, **kwargs):
        """
        returns the matrix decomposition of the chosen kwarg 
        kwarg to choose decomp 
        ex 
        svd=dict
        dict = {
            method:tsvd
            ncomponents:22
        }
        returns the 22 component decomp
        """
        def count_digits(n):
            if n > 0:
                digits = int(math.log10(n))+1
            elif n == 0:
                digits = 1
            elif n < 0:
                digits = int(math.log10(-n))+2
            return digits
        
        keys = kwargs.keys()
        if "tsvd" in keys:
            method = kwargs["tsvd"]
            method_keys = kwargs["tsvd"].keys()
            if "n_components" in method_keys:
                n_components = method["n_components"]
                if n_components <= min(X.shape[0], X.shape[1]):
                    U, s, Vh = svds(X, k=n_components) # Compute the truncated svd out to k components
                    return U, s, Vh
                else:
                    print("n_components exceeds the min of column or row...must be less than ", min(X.shape[0], X.shape[1]) )

        if "snf" in keys:
            method = kwargs["snf"]
            method_keys = kwargs["snf"].keys()
            scaling = math.pow(10, count_digits(np.min( X[np.nonzero(X)] )  ) )
            m, n = X.shape[0:2]
            kkmatrix = scaling*X
            elements = [z.Z(int(ele)) for ele in kkmatrix.reshape(m*n) ]
            kkmatrix = matrix.Matrix(h=m, w=n, elements=elements)

            snfp = snfproblem.SNFProblem(kkmatrix)
            snfp.computeSNF()
            print(snfp.isValid())
            if snfp.isValid():
                A, S, J, T = snfp.A, snfp.S, snfp.J, snfp.T
                A = np.asarray([ele.a for ele in A.elements[:]] ).reshape(m,n)
                S = np.asarray([ele.a for ele in S.elements[:]]).reshape(m,n)
                T = np.asarray([ele.a for ele in T.elements[:]]).reshape(m,n)
                J = np.asarray([ele.a for ele in J.elements[:]]).reshape(m,n)
                return np.linalg.inv(S), J, np.linalg.inv(T)
            print("snfp is not valid")
        
    @staticmethod
    def freq_kernel(psf, X):
        def pad_with_zeros(vec, pad_width, iaxis, kwargs):
            vec[:pad_width[0]] = 0
            vec[-pad_width[1]:] = 0
            return vec
        """
        returns a frequency shifted fft kernel
        """
        #kernel = np.pad(psf, (((X.shape[0] - 3)//2, (X.shape[0] - 3)//2 + 1)), pad_with_zeros  )
        kernel = np.pad(psf, (((X.shape[0] - psf.shape[0])//2, (X.shape[0] - psf.shape[1])//2 + 1)), pad_with_zeros  )
        freq_kernel = np.fft.fft2( np.fft.ifftshift(kernel) )
        return freq_kernel

    
    @staticmethod
    def sub_image(X, d):
        Queue = []
        str = f"is {X.shape[0]} mod {d} = 0 ... {X.shape[0]%d==0} \n  is {X.shape[1]} mod {d} = 0 ... {X.shape[1]%d==0}"
        print(str)
        try:
            v_split = np.vsplit(X, d)
            particians = [np.hsplit(ele, d) for ele in v_split ]
        except ValueError:
            return Queue

        for element in particians:
            for ele in element:
                Queue.append(ele)
        print("shape of sub image is ", ele.shape )
        return Queue

    @staticmethod
    def paste_image(que, d):
        try:
            que = np.asarray(que)
            v_split = np.vsplit(que, d)
            partician = [ np.hstack(ele) for ele in v_split]
            return np.vstack(partician)
        except ValueError as e:
            print(e)
            print("Value Error")

    @staticmethod
    def confidence_intervals(X, img, alpha, n , **kwargs):
        keys = kwargs.keys()
        gamma = np.zeros(img.shape)
        sum_x_sq = np.zeros(img.shape)
        if "nlsq" in keys and kwargs["nlsq"]:
            #add noise
            X.noise_img(mean_noise=0, sigma_noise= X.sigma_noise)
            # compute recovered image
            param = {        
                "sigma_blur" : X.sigma_blur,
                "ksize": X.kernel_size, 
                "K": Image_Processing.blur_matrix(X.psf, img.shape )
            }
            for trial in range(n):
                param['X_0'] =  Image_Processing.first_approximation(img, X.psf, clip=True)
                param["Y"]  = img
                img = Image_Processing.restore(param=param, nlsq=True)
                gamma += img
                sum_x_sq += img*img
            mean_img =  gamma/(trial-2)
            var = np.abs(sum_x_sq/(trial-2) - mean_img*mean_img)

            mean_img = mean_img.reshape(gamma.shape)
            interval = st.norm.interval(alpha=alpha,
                                        loc=mean_img,
                                        scale = np.sqrt(var)
                                        )            
                    
        if "naive" in keys and kwargs["naive"]:
            #add noise
            X.noise_img(mean_noise=0, sigma_noise= X.sigma_noise)
            # compute recovered image
            param = {        
                "sigma_blur" : X.sigma_blur,
                "ksize": X.kernel_size, 
                "K": Image_Processing.blur_matrix(X.psf, img.shape ).astype(np.float64)
            }
            for trial in range(n):
                param["Y"]  = img
                img = Image_Processing.restore(param=param, naive=True)
                gamma += img
                sum_x_sq += img*img
            mean_img =  gamma/(trial)
            var = np.abs(sum_x_sq/(trial-2) - mean_img*mean_img)

            interval = st.norm.interval(alpha=alpha,
                                            loc=mean_img,
                                            scale = np.sqrt(var)
                                            )   
            #interval = st.norm.interval(alpha= alpha,
            #                loc = mean_img,
            #                scale = st.sem(mean_img,
            #                nan_policy='omit' ))
            #                
        return np.array(interval)

    @staticmethod
    def restore(param, **kwargs):
        keys = kwargs.keys()
        p_key = param.keys()
        if "nlsq" in keys and kwargs["nlsq"]:
            res_img = param["Y"]
            img = param["X_0"]

            sigma_b = param["sigma_blur"]
            c = 1
            try:
                m, n, c = res_img.shape
            except ValueError as e:
                m, n = res_img.shape

            def IP_blur(x, sigma_b=sigma_b, ksize=param["ksize"], res_img=res_img):
                c = 1
                try:
                    m, n, c = res_img.shape
                    x = x.reshape(m, n, c)
                except ValueError as e:
                    m, n = res_img.shape
                    x = x.reshape(m, n)

                blur = cv2.GaussianBlur(x, ksize=ksize, sigmaX=sigma_b, sigmaY=sigma_b, borderType=cv2.BORDER_CONSTANT)
                return  blur.reshape(m*n*c, ) - res_img.reshape(m*n*c, )

            input = img.reshape(m*n*c, )
            res = least_squares(IP_blur, input, bounds=(0.0, 255.0))
            
            #res.x =  255*( res.x /np.max(res.x))

            if c == 1:
                return res.x.reshape(m, n)
            else:
                return res.x.reshape(m, n, c)

        if "naive" in keys and kwargs["naive"]:
            img = param["Y"]
            k = param["K"]
            vec = img.reshape(img.size)
            y = (np.linalg.inv(k)@vec)
            y = 255*(np.abs(y)/np.max(y))
            return y.reshape(img.shape)


    @staticmethod
    def first_approximation(img, psf,**kwargs):
        TOL = 0.000001
        keys = kwargs.keys()
        img_f = np.fft.fft2(img)
        img_f = img_f
        h_f = Image_Processing.freq_kernel(psf, img)
        #h_f = h_f/np.max(h_f)
        img =   np.abs(np.fft.ifft2(img_f/h_f) ).astype(dtype=np.float32)
        #if img_f.shape == h_f.shape:
        #    for i in range(h_f.shape[0]):
        #        for j in range(h_f.shape[1]):
        #            if h_f[i,j] >= TOL:
        #                img_f[i,j] = img_f[i,j]/h_f[i,j]
        #            else:
        #                img_f[i, j] = 255
        #img = np.abs(np.fft.ifft2(img_f)).astype(dtype=np.float32)


        A = img
        if "clip" in keys and kwargs["clip"]:
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if img[i,j] < 0.0:
                        A[i,j] = 0.0
                        #print(Image_Processing.distortion(img, A))
                    if img[i,j] > 255.0:
                        A[i,j] = 255.0
                        #print(Image_Processing.distortion(img, A))
        return A

    @staticmethod
    def plot_confidence(img, LI, UI,XT, fname, **kwargs):
        keys = kwargs.keys()
        BLUE = img.reshape(img.size)
        lb = LI.reshape(LI.size)
        ub = UI.reshape(UI.size)
        xt = XT.reshape(XT.size)
        df = pd.DataFrame({"lower bound": np.transpose(lb),
                            "BLUE": np.transpose(BLUE),
                            "Upper bound": np.transpose(ub),
                            "True Image": np.transpose(xt)},
                            columns=["lower bound", "BLUE", "Upper bound", "True Image"])

        df = df.applymap(Image_Processing.clip)
        if "kde" in keys and kwargs["kde"]:
            df.plot(kind="kde", title=f"Reconstruction using {fname} model")
            Photo.show_array(np.hstack([LI, img, UI]), title=f"Reconstruction using {fname} model")
            return df

        if "exact" in keys and kwargs["exact"]:
            df.reset_index()
            for col in df.columns:
                plt.scatter(df.index, df[col], s=1 )
            plt.legend(df.columns)
            Photo.show_array(np.hstack([LI, img, UI]))
            #plt.savefig(fname +"Confidence_interval.png")
            return df


    @staticmethod
    def clip( element):
        if 0 <= element and element <=255:
            return element
        elif element <=0:
            return 0
        elif element > 255:
            return 255


