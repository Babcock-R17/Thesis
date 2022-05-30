# testing model for multiplicitive noise 



# Imports

import matplotlib
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


def tsvd(img, k_matrix, n_components, **kwargs):
    def m_mat_over_channels(A, img):
        image_shape = img.shape
        m, n = image_shape[0:2]
        for ch in range(len( img.shape )):
            xvec = img[:, :, ch].reshape( m*n, 1 )
            img_temp = (A@xvec)
            img[:, :, ch] = img_temp.reshape( (image_shape[0:2]) )
        return img
    
    keys = kwargs.keys()

    image_shape = img.shape 
    if "deblur" in keys and kwargs["deblur"]:
        #U, s, Vh = svds(k_matrix, k=n_components) # Compute the truncated svd out to k components
        U, s, Vh = np.linalg.svd(k_matrix, full_matrices=True) # Compute the full svd  
        if "multi_noise" in keys:
            delta = kwargs['multi_noise']
            s = s*delta[0:len(s)]
            s = s/max(s)
        k_inv = np.transpose(Vh) @np.diag(1/s) @ np.transpose(U)
        return m_mat_over_channels( k_inv, img )
    
    elif "blur" in keys and kwargs["blur"]:
        U, s, Vh = np.linalg.svd(k_matrix, full_matrices=True) # Compute the full svd  
        if "multi_noise" in keys:
            delta = kwargs['multi_noise']
            s = s*delta[0:len(s)]
            s = s/max(s) 
        k = U@np.diag(s)@Vh
        return m_mat_over_channels( k, img )





def LDE(*values, **kwargs):
    gcd = math.gcd(*values)
    print(kwargs.keys)
    print(gcd)
    print( kwargs['value'] % gcd)
    if kwargs['value'] % gcd == 0:
        return True
    return False



def EE_gcd(a, b):
    """
    Returns a list `result` of size 3 where:
    Referring to the equation ax + by = gcd(a, b)
        result[0] is gcd(a, b)
        result[1] is x
        result[2] is y 
    """
    s = 0; old_s = 1
    t = 1; old_t = 0
    r = b; old_r = a

    while r != 0:
        quotient = old_r//r # In Python, // operator performs integer or floored division
        # This is a pythonic way to swap numbers
        # See the same part in C++ implementation below to know more
        old_r, r = r, old_r - quotient*r
        old_s, s = s, old_s - quotient*s
        old_t, t = t, old_t - quotient*t
    return [old_r, old_s, old_t]

def test_model(image_selection, results, paramaters):
    # Setting Expected key argurments
    fp = image_selection['fp']
    fname =  image_selection['fname']
    color_map = 'viridis'
    title = "Title: "+ fname[0:-4]
    save_at = results['fp']
    save_as = results['fname']
    style='seaborn-talk'


    def create_save(prop="_", save_at = results['fp'], save_as = results['fname']):
        return  os.path.join(save_at, str(prop )+"_" + save_as )

    # Loading desired image X as a Photo object.
    X = P.load(fp=fp, fname=fname)
    # Cropping Photo to explore reasonable sized matrices
    X = P.transform(X, crop='0:20, 0:20'  )
    X = ip(fp=fp, fname=fname, img=X )  # Set X as an image_processing class


    # Parameters are 
    PSF_Size = paramaters["PSF_Size"]
    NS_lvl = paramaters["NS_lvl"]
    BS_lvl = paramaters['BS_lvl']

    # Setting properties of image X
    X.blur_img(ksize=PSF_Size, sigma_blur=BS_lvl)
    X.noise_img(mean_noise=0, sigma_noise=NS_lvl)

    # Getting associated properties.... 
    k = X.blur_matrix
    df_kmat = pd.DataFrame(X.blur_matrix, dtype=np.float64)

    def blur_ch(k, img):
        image_shape = img.shape
        m, n = image_shape[0:2]
        for ch in range(len( img.shape )):
            img[:, :, ch] = ( k@img[:, :, ch].reshape( m*n, 1 ) ).reshape( (m, n) )
        return img

    X_blur = blur_ch(k, X.image)
    # P.show_array(X_blur, title="title: My blur ")
    # X.blur.show(title="title: Built in blur")

    # Compare both versions of blurring
    X_blur = ip(fp=fp, fname="My_Blurring_on_"+fname, img=X_blur)
    built_blur =  ip(fp=fp, fname= "Pre_Built_Blurring_on_"+ fname, img= X.blur.image )


    X_bf = X_blur.fftn()
    bb_f = built_blur.fftn()

    # comparing 
    comparison = X_bf/bb_f
    checks = abs(sp.fft.ifftn(comparison ) )
    checks = 255*checks.astype(np.uint8)

    #ip.display(checks, title=" comparison on built in method ", manifold=True)


    '''
    method for blurring matrix only works if i specify a square image. 
    and 
    the size is relatively small
    '''

def modelx(image_selection, results, paramaters):
    # Setting Expected key argurments
    fp = image_selection['fp']
    fname =  image_selection['fname']
    color_map = 'viridis'
    title = "Title: "+ fname[0:-4]
    save_at = results['fp']
    save_as = results['fname']
    style='seaborn-talk'


    def create_save(prop="_", save_at = results['fp'], save_as = results['fname']):
        return  os.path.join(save_at, str(prop )+"_" + save_as )
    

    # Loading desired image X as a Photo object.
    X = P.load(fp=fp, fname=fname)
    # Cropping Photo to explore reasonable sized matrices
    X = P.transform(X, crop='0:4, 0:4'   )
    X = ip(fp=fp, fname=fname, img=X )  # Set X as an image_processing class
    title=title+"Original size {}".format(X.shape )
    X.show(title=title)


    # Parameters are 
    PSF_Size = paramaters["PSF_Size"]
    NS_lvl = paramaters["NS_lvl"]
    BS_lvl = paramaters['BS_lvl']

    # Setting properties of image X
    X.blur_img(ksize=PSF_Size, sigma_blur=BS_lvl)
    X.noise_img(mean_noise=0, sigma_noise=NS_lvl)

    # Getting associated properties.... 
    k_matrix = ip.blur_matrix(X.psf, X.shape)
    kdf = pd.DataFrame(k_matrix)
    at_row = 4

    value = X.noise.image[at_row%3, at_row//3]



    # lets view the possible truncated SVD 

    n_components = min(k_matrix.shape ) - 100

    print("SVD Reconstruction using: ", n_components)

    from scipy.sparse.linalg import svds
    #U, s, Vh = svds(k_matrix, k=n_components) # Compute the truncated svd out to k components
    #Uh = np.transpose(U)
    #1. Simulate an image that has blur and multiplicive noise
    # 
    m, n = X.image.shape[0:2]
    delta = np.random.normal(X.mean_noise, X.sigma_noise, m*n) + 1 
    print(delta)
    # simulation of blur process when there is multiplicative noised involved
    # View each blur step first. 
    # 
    
    

    #blur_temp = tsvd(img=X.image, k_matrix=k_matrix, n_components=n_components, blur=True)
    xtest = tsvd(img=X.blur.image,  k_matrix=k_matrix, n_components=n_components, deblur=True)
    P.show_array(xtest, title="Blurred Deblurred using svd with "+ str(n_components)+" components")
    print('distortion is {}'.format(ip.distortion(X.image, xtest) ))


    xtest = tsvd(img=X.blur.image + X.noise.image,  k_matrix=k_matrix, n_components=n_components, deblur=True)
    P.show_array(xtest, title="blurred and Noised Deblurred using svd with "+ str(n_components)+" components")
    print('distortion is {}'.format(ip.distortion(X.image, xtest) ))


def snf(image_selection, results, paramaters):
    def m_mat_over_channels(A, img):
        image_shape = img.shape
        m, n = image_shape[0:2]
        for ch in range(len( img.shape )):
            xvec = img[:, :, ch].reshape( m*n, 1 )
            img_temp = (A@xvec)
            img[:, :, ch] = img_temp.reshape( (image_shape[0:2]) )
        return img

    from smithnormalform import matrix, snfproblem, z

    # Setting Expected key argurments
    fp = image_selection['fp']
    fname =  image_selection['fname']
    color_map = 'viridis'
    title = "Title: "+ fname[0:-4]
    save_at = results['fp']
    save_as = results['fname']
    style='seaborn-talk'


    def create_save(prop="_", save_at = results['fp'], save_as = results['fname']):
        return  os.path.join(save_at, str(prop )+"_" + save_as )
    

    # Loading desired image X as a Photo object.
    X = P.load(fp=fp, fname=fname)
    # Cropping Photo to explore reasonable sized matrices
    X = P.transform(X, crop='0:4, 0:4'   )
    X = ip(fp=fp, fname=fname, img=X )  # Set X as an image_processing class
    title=title+"Original size {}".format(X.shape )
    X.show(title=title)


    # Parameters are 
    PSF_Size = paramaters["PSF_Size"]
    NS_lvl = paramaters["NS_lvl"]
    BS_lvl = paramaters['BS_lvl']

    # Setting properties of image X
    X.blur_img(ksize=PSF_Size, sigma_blur=BS_lvl)
    X.noise_img(mean_noise=0, sigma_noise=NS_lvl)

    # Getting associated properties.... 
    k_matrix = ip.blur_matrix(X.psf, X.shape)

    scaling = 1/np.min(k_matrix[np.nonzero(k_matrix)] ) 
    m, n = k_matrix.shape[0:2]
    kkmatrix = scaling*k_matrix
    elements = [z.Z(int(ele)) for ele in kkmatrix.reshape(m*n) ]
    kkmatrix = matrix.Matrix(h=m, w=n, elements=elements)

    snfp = snfproblem.SNFProblem(kkmatrix)
    snfp.computeSNF()
    print(snfp.isValid())
    A, S, J, T = snfp.A, snfp.S, snfp.J, snfp.T
    A = np.asarray([ele.a for ele in A.elements[:]] ).reshape(m,n)
    S = np.asarray([ele.a for ele in S.elements[:]]).reshape(m,n)
    T = np.asarray([ele.a for ele in T.elements[:]]).reshape(m,n)
    J = np.asarray([ele.a for ele in J.elements[:]]).reshape(m,n)

    J_hinv = np.where(J!=0, scaling*1/J, 0)


    U, s, Vh = np.linalg.svd(S)
    k_invp = T@J_hinv@ U@ np.diag(s)@Vh 

    B = X.blur.image
    Xpid = m_mat_over_channels( k_invp, B) 
    Xpid = Xpid%255
    P.show_array(Xpid, title="Xpid solution has percent error of "+ str(ip.distortion(Xpid, X.image) ) )
    print("Xpid distortion is ", ip.distortion(Xpid, X.image) )


    BN = X.blur.image + X.noise.image
    Xpid = m_mat_over_channels( k_invp, BN) 
    Xpid = Xpid%255
    P.show_array(Xpid, title="Xpid blurred and noised solution has percent error of "+ str(ip.distortion(Xpid, X.image) ) )
    print("Xpid distortion is ", ip.distortion(Xpid, X.image) )


    #





    







if __name__ == "__main__":
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



    fname = "interlaced_0126.jpg"
    image_selection={
        'fp': image_path,
        'fname': fname
        }

    results = {
        'fp': os.path.join(workspace_path, "Images", "interlaced"),
        'fname': fname
    }

    paramaters = { 
        "PSF_Size": (3 , 3 ),
        "NS_lvl": 0.05, 
        "BS_lvl": 11
        } 

    #test_model( image_selection=image_selection, results=results, paramaters=paramaters  )
    #modelx(image_selection, results, paramaters)
    #snf(image_selection, results, paramaters)


