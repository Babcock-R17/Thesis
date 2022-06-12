# testing model for multiplicitive noise 



# Imports
import matplotlib
from matplotlib.colors import rgb2hex
from matplotlib.pyplot import specgram
from Photo import Photo as P
from Image_Processing import Image_Processing as ip
import os
from os import listdir
import numpy as np
import pandas as pd
import scipy as sp

## 
from scipy.sparse.linalg import svds
import math

def set_image():
    # Setting up the paramaters 
    # Assumptions: Noise is i.i.d and Gaussian noise 
    #              Blur Matrix, blurring_sigma, are Known
    #              

    # Load an image and Describe ways of viewing an image.
    # Define File Paths
    workspace_path = "Thesis_base_level_1"
    water_path = os.path.join(workspace_path, "Watermarks")
    image_path = os.path.join(workspace_path, "Custom")
    image_names = listdir(image_path)
    watermark_names = listdir(water_path)



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
        "PSF_Size": (3 , 3 ),
        "NS_lvl": 0.05, 
        "BS_lvl": 11
        } 
    # Setting Expected key argurments
    fp = image_selection['fp']
    fname =  image_selection['fname']
    #color_map ='viridis'
    #title = "Title: "+ fname[0:-4]
    #save_at = results['fp']
    #save_as = results['fname']
    #style='seaborn-talk'
    def create_save(prop="_", save_at = results['fp'], save_as = results['fname']):
        return  os.path.join(save_at, str(prop )+"_" + save_as )
    
    # Loading desired image X as a Photo object.
    X = P.load(fp=fp, fname=fname)
    # Cropping Photo to explore reasonable sized matrices
    shape = X.shape
    db = 5 # pixels to crop a boarder
    ROI = f"{0+db}:{shape[0] - db},{0+db}:{shape[1] - db}"
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

def visualizations(X, show=True):
    style='seaborn-talk'
    color_map ='viridis'
    
    X.show(title=f"Original Image\nDimensions {X.shape}")
    '''
    ip.display(X.image, title=f"Original Image\nDimensions {X.shape}",
                manifold=True,
                animate=True,
                show=show,
                fname="manifold",
                figsize=(10, 10),
                style=style,
                color_map=color_map)

    ip.display(X.blur.image, title=f"BLURRED Original Image\nDimensions {X.shape}",
                manifold=True,
                animate=True,
                show=show,
                fname="blur_manifold",
                figsize=(10, 10),
                style=style,
                color_map=color_map)

    ip.display(X.noise.image, title=f"Gaussian Noise Image\nDimensions {X.shape}",
                manifold=True,
                animate=True,
                show=show,
                fname="noise_manifold",
                figsize=(10, 10),
                style=style,
                color_map=color_map)

    ip.display(X.blur.image + X.noise.image, title=f"Gaussian Blurred and Noised Image\nDimensions {X.shape}",
                manifold=True,
                animate=True,
                show=show,
                fname="Blurred_noised_manifold",
                figsize=(10, 10),
                style=style,
                color_map=color_map)



    ip.display(X.image, title=f"Original Image\nDimensions {X.shape}",
                signals=True,
                animate=True,
                show=show,
                fname="Signals",
                figsize=(10, 10),
                style=style,
                color_map=color_map)

    '''



    ip.display(X.blur.image, title=f"BLURRED Original Image\nDimensions {X.shape}",
                signals=True,
                animate=True,
                show=show,
                fname="blur_Signals",
                figsize=(10, 10),
                style=style,
                color_map=color_map)

    ip.display(X.noise.image, title=f"Gaussian Noise Image\nDimensions {X.shape}",
                signals=True,
                animate=True,
                show=show,
                fname="noise_Signals",
                figsize=(10, 10),
                style=style,
                color_map=color_map)

    ip.display(X.blur.image + X.noise.image, title=f"Gaussian Blurred and Noised Image\nDimensions {X.shape}",
                signals=True,
                animate=True,
                show=show,
                fname="Blurred_noised_Signals",
                figsize=(10, 10),
                style=style,
                color_map=color_map)

def visualize_psf(X):
    def pad_with_zeros(vec, pad_width, iaxis, kwargs):
        vec[:pad_width[0]] = 0
        vec[-pad_width[1]:] = 0
        return vec

    psf = X.psf
    X = P.transform(X.image, rgb2gray=True )
    style='seaborn-talk'
    color_map ='viridis'


    H = ip.freq_kernel(psf, X)
    H_f = np.fft.ifftshift(H)
    H_freq = np.abs(np.fft.ifftshift(H_f)).astype(np.float32)

    # Shift zero-frequency component to the center of the spectrum.
    X_f = np.fft.fft2( np.fft.ifftshift(X) )
    m, n = H_f.shape

    print(X.shape)
    blur_f = np.zeros(X.shape, dtype=np.complex128)

    if len(X.shape) == 3:
        m1, n1, c = X.shape
        for ch in range(c):
            blur_f[:min(m,m1), : min(n,n1) ,ch] = H_freq[:min(m,m1), : min(n,n1)  ]*X_f[:min(m,m1), :min(n,n1), ch ]
    else:
        m1, n1 = X.shape
        blur_f[:min(m,m1), : min(n,n1)] = H_freq[:min(m,m1), : min(n,n1)  ]*X_f[:min(m,m1), :min(n,n1) ]



    blur = np.abs( np.fft.ifft2(np.fft.ifftshift(blur_f) )  ).astype(np.float64)
    blur = 255*blur/np.max(blur)

    

    H  =   np.abs(np.fft.ifft2(H_freq )).astype(np.float64)
    P.show_array(blur)

    print( H.shape, blur.shape, H.dtype , blur.dtype )

    ip.display(  H , title=f"PSF as a Signal\nDimensions {psf.shape}",
                signals=True,
                animate=False,
                show=False,
                fname="PSF as a Signal",
                figsize=(10, 10),
                style=style,
                color_map=color_map)


    ip.display(blur, title=f"Resulting Product as a Signal\nDimensions {blur_f.shape}",
            signals=True,
            animate=False,
            fname="(PSF product with Image)",
            figsize=(10, 10),
            style=style,
            color_map=color_map)

    pass


def main():
    X = set_image()
    # Visualize the Image and create gifs for the slides
    visualizations(X, show=False)
    # Done

    # visualize the PSF 
    #visualize_psf(X)

    # Set up visualizations for problem statement
    # i.e. subimages
    #ip.confidence_intervals(X, method="model_CI", ntrials=10)
    

    # partician image 
    # 
    # 







    pass







if __name__ == "__main__":
    main()





























