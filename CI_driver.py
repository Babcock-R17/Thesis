__author__ = "Jasen Babcock"


from Photo import Photo as P
from Image_Processing import Image_Processing as ip
import os
from os import listdir
import numpy as np
import matplotlib.pyplot  as plt
import pandas as pd


def set_image():
    # Setting up the paramaters 
    # Assumptions: Noise is i.i.d and Gaussian noise 
    #              Blur Matrix, blurring_sigma, are Known
    #              

    # Load an image and Describe ways of viewing an image.
    # Define File Paths
    workspace_path = "Thesis_base_level_1"
    image_path = os.path.join(workspace_path, "Custom")



    fname = "cat.jpg"
    image_selection={
        'fp': image_path,
        'fname': fname
        }

    results = {
        'fp': os.path.join(workspace_path, "Images", "cat"),
        'fname': fname
    }

    paramaters = { 
        "PSF_Size": (151 , 151 ),
        "NS_lvl": 1.10, 
        "BS_lvl": 10*1.10
        } 
    # Setting Expected key argurments
    fp = image_selection['fp']
    fname =  image_selection['fname']

    def create_save(prop="_", save_at = results['fp'], save_as = results['fname']):
        return  os.path.join(save_at, str(prop )+"_" + save_as )
    
    # Loading desired image X as a Photo object.
    X = P.load(fp=fp, fname=fname)
    # Cropping Photo to explore reasonable sized matrices

    def calc_ROI(shape, gcd=2**4):
        m, n = (shape[0]%gcd)//2, (shape[1]%gcd)//2
        print(m,n)
        ROI = f"{0+m}:{shape[0] - m},{0+n}:{shape[1] - n}"
        return ROI

    shape = X.shape
    ROI = f"{240 }:{ 1040},{560 }:{1360}"
    print(ROI)



    X = P.transform(X, crop=ROI, rgb2gray=True   )
    X = ip(fp=fp, fname=fname, img=X )  # Set X as an image_processing class
    print(f"set PSF_Size {paramaters['PSF_Size']}\n Set Blurring Level {paramaters['NS_lvl']}\n Set Noise Level {paramaters['BS_lvl']}")
    
    # Setting properties of image X
    # Parameters are 
    PSF_Size = paramaters["PSF_Size"]
    NS_lvl = paramaters["NS_lvl"]
    BS_lvl = paramaters['BS_lvl']

    X.blur_img(ksize=PSF_Size, sigma_blur=BS_lvl)
    X.noise_img(mean_noise=0, sigma_noise=NS_lvl)
    return X


def set_small_image():
    # Setting up the paramaters 
    # Assumptions: Noise is i.i.d and Gaussian noise 
    #              Blur Matrix, blurring_sigma, are Known
    #              

    # Load an image and Describe ways of viewing an image.
    # Define File Paths
    workspace_path = "Thesis_base_level_1"
    image_path = os.path.join(workspace_path, "Custom")



    fname = "cat.jpg"
    image_selection={
        'fp': image_path,
        'fname': fname
        }

    results = {
        'fp': os.path.join(workspace_path, "Images", "cat"),
        'fname': fname
    }

    paramaters = { 
        "PSF_Size": (15 , 15 ),
        "NS_lvl": 1.10, 
        "BS_lvl": 1.10
        } 
    # Setting Expected key argurments
    fp = image_selection['fp']
    fname = image_selection['fname']

    
    # Loading desired image X as a Photo object.
    X = P.load(fp=fp, fname=fname)
    # Cropping Photo to explore reasonable sized matrices
    ROI = f"{590 }:{ 690},{910 }:{1010}"
    print(ROI)



    X = P.transform(X, crop=ROI, rgb2gray=True   )
    X = ip(fp=fp, fname=fname, img=X )  # Set X as an image_processing class
    X.show()
    print(f"set PSF_Size {paramaters['PSF_Size']}\n Set Blurring Level {paramaters['NS_lvl']}\n Set Noise Level {paramaters['BS_lvl']}")
    
    # Setting properties of image X
    # Parameters are 
    PSF_Size = paramaters["PSF_Size"]
    NS_lvl = paramaters["NS_lvl"]
    BS_lvl = paramaters['BS_lvl']

    X.blur_img(ksize=PSF_Size, sigma_blur=BS_lvl)
    X.noise_img(mean_noise=0, sigma_noise=NS_lvl)
    return X


def slide_1(X):
    """
    slide to show the partician process
    confidence intervals of the restoration
    Method Naive
    """
    

    X.show(title=f"Original Image {X.fname } \n Dimension {X.shape}")
    
    # size is set to 800 by 800
    d = 100
    #queue = ip.sub_image(X.blur.image+ X.noise.image, d)
    queue = ip.sub_image(X.blur.image, d)
    
    
    paramaters = {
        "sigma_blur" : X.sigma_blur,
        "ksize": X.kernel_size
        }


    restored = []
    for img in queue:
        k = ip.blur_matrix(X.psf, img.shape )
        paramaters["Y"] = img
        paramaters["X_0"] = ip.first_approximation(img=img, psf=X.psf, clip=True)
        paramaters["k"] = k
        #partician = ip.restore(param=paramaters, nlsq=True) 
        #restored.append( partician   )
        restored.append( (np.linalg.inv(k)@img.reshape(img.size) ).reshape(img.shape) )
    
    
    print(f"loaded...{np.asarray(restored).shape}" )

    X = ip.paste_image(restored, d)


    style='seaborn-talk'
    color_map ='viridis'
    #color_map ='gray'

    t1 =f"""True B.L.U.E Image\n
                Method: Naive\n
                Particians: {int(np.sqrt(len(queue)))} {queue[0].shape} Images\n
                Dimension: {X.shape}
                """

    ip.display(X ,
        title=t1,
        manifold=True,
        animate=True,
        show=False,
        fname="Naive Reconstruction of Image",
        figsize=(10, 10),
        style=style,
        color_map=color_map)


    ip.display(X ,
        title=t1 ,
        signals=True,
        animate=True,
        show=False,
        fname="Naive Reconstruction as Signals",
        figsize=(10, 10),
        style=style,
        color_map=color_map)

    """
    Caption:
    Theoretical Image reconstruction
    K_inv Y as sub images
    Gaussian Blur: 3%
    Gaussian Noise: 5%     

        "PSF_Size": (3 , 3 ),
        "NS_lvl": 0.05, 
        "BS_lvl": 0.03
    """

def slide_3(X,fname="Naive"):
    """
    confidence intervals of the restoration
    Method Naive
    """
    style='seaborn-talk'
    color_map ='viridis'   

    #X = set_small_image()
    X.show(title=f"Cat Cropped\nDimension: {X.shape}")
    d = 100
    queue = ip.sub_image(X.blur.image+ X.noise.image, d)
    paramaters = {
        "sigma_blur" : X.sigma_blur,
        "ksize": X.kernel_size
        }
    lower_img = []
    upper_img = []
    for img in queue:
        k = ip.blur_matrix(X.psf, img.shape )
        paramaters["Y"] = img
        paramaters["k"] = k
        LI , UI = ip.confidence_intervals(X, img, alpha=0.90, n=31, naive=True)
        lower_img.append(LI.reshape(img.shape))
        upper_img.append(UI.reshape(img.shape))

    
    print(f"loaded...{np.asarray(lower_img).shape} with method {fname}" )

    X_L = ip.paste_image(lower_img, d)
    X_U = ip.paste_image(upper_img, d)
    X =  0.5*(X_L + X_U)
    
    P.show_array(X_L, title="lower bound")
    P.show_array(X, title="BLUE")
    P.show_array(X_U, title="Upper bound")

    ip.plot_confidence(X, X_L, X_U, kde=True, fname=fname)

    t1 =f"""True B.L.U.E Image\n
                Method: Nonlinear least squares\n
                Particians: {int(np.sqrt(len(queue)))} {queue[0].shape} Images\n
                Dimension: {X.shape}
                """

    ip.display(X ,
        title=t1,
        manifold=True,
        animate=True,
        show=False,
        fname="NLSQ Reconstruction of Image",
        figsize=(10, 10),
        style=style,
        color_map=color_map)


    ip.display(X ,
        title=t1 ,
        signals=True,
        animate=True,
        show=False,
        fname="NLSQ Reconstruction as Signals",
        figsize=(10, 10),
        style=style,
        color_map=color_map)

def slide_4(X, fname="nlsq"):
    """
    confidence intervals of the restoration
    Method Bounded Non Linear Least Squares
    """
    style='seaborn-talk'
    color_map ='viridis'   

    #X = set_small_image()
    X.show(title=f"Cat Cropped\nDimension: {X.shape}")
    d = 100
    queue = ip.sub_image(X.blur.image+ X.noise.image, d)
    paramaters = {
        "sigma_blur" : X.sigma_blur,
        "ksize": X.kernel_size
        }

    lower_img = []
    upper_img = []
    for img in queue:
        k = ip.blur_matrix(X.psf, img.shape )
        paramaters["Y"] = img
        paramaters["X_0"] = ip.first_approximation(img=img, psf=X.psf, clip=True)
        paramaters["k"] = k
        LI , UI = ip.confidence_intervals(X, img, alpha=0.90, n=31, nlsq=True)
        lower_img.append(LI.reshape(img.shape))
        upper_img.append(UI.reshape(img.shape))

    
    print(f"loaded...{np.asarray(lower_img).shape} with method {fname}" )

    X_L = ip.paste_image(lower_img, d)
    X_U = ip.paste_image(upper_img, d)
    X =  0.5*(X_L + X_U)
    
    P.show_array(X_L, title="lower bound")
    P.show_array(X, title="BLUE")
    P.show_array(X_U, title="Upper bound")

    ip.plot_confidence(X, X_L, X_U, kde=True, fname=fname)

def section_ci():
    """
    running section Confidence Intervals...
    """
    X = set_small_image()
    print( section_ci.__doc__)

    def full_to_small():
        """
        run to describe cropping a large image and working on a small scale image...
        """
        print(full_to_small.__doc__)
        X = set_image()
        t_0 = f"Original Image {X.shape} "
        P.show_array(X.image, title=t_0)
        t_1 = f"Blurred Image {X.shape}\nBlurr: {X.sigma_blur}%\nPSF shape: {X.psf.shape} "
        P.show_array(X.blur.image, t_1)
        t_2 = f"Blurred, Noisy Image {X.shape} \nBlurr:{X.sigma_blur}%\nPSF shape:{X.psf.shape}\nNoise: {X.sigma_noise}%"
        P.show_array(X.blur.image + X.noise.image, t_2)

        X = set_small_image()
        t_0 = f"Original Image {X.shape} "
        P.show_array(X.image, title=t_0)
        t_1 =f"Blurred Image {X.shape}\nBlurr:{X.sigma_blur}%\nPSF shape:{X.psf.shape} "
        P.show_array(X.blur.image, t_1)
        t_2 = f"Blurred, Noisy Image {X.shape} \nBlurr: {X.sigma_blur}%\nPSF shape: {X.psf.shape}\nNoise: {X.sigma_noise}%"
        P.show_array(X.blur.image + X.noise.image, t_2)

        pass


    full_to_small()



    pass






def main():
    #X = set_image()
    #slide_1(X)
    #slide_3(X)
    #slide_4(X)
    section_ci()






if __name__ == "__main__":
    main()


