# Driver for Thesis
from Photo import Photo
from Image_Processing import Image_Processing

import os
from os import listdir
import streamlit as st
import scipy as sp
import numpy as np
import cv2
from scipy.optimize import least_squares



def main():
    print("running main...")
    # Define File Paths
    workspace_path = "Independent_Study"
    water_path = os.path.join(workspace_path, "Watermarks")
    image_path = os.path.join(workspace_path, "Custom")
    image_names = listdir(image_path)
    watermark_names = listdir(water_path)
    print("Loaded {} \n Loaded {}".format(image_names, watermark_names))

    # Helper functions
    def update_blend(**kwargs1):
        selected_image = kwargs1[selected_image]
        selected_watermark=kwargs1[selected_watermark]
        scale=kwargs1[scale]
        location=kwargs1[location]
        alpha=kwargs1[alpha]
        return selected_image.alpha_blend_watermark(selected_watermark.image, scale, location, alpha )

    def select_target(res_img, img):
        ## initialize each channel
        ks=0
        sigma_blur=1
        mean_noise=1
        ks=st.slider("ksize = 2(ks) + 1",
            min_value=0,
            max_value=10,
            step=1)
        sigma_blur = st.slider("sigma_blur",
            min_value=0,
            max_value=10,
            step=1)
        ksize = (2*ks + 1, 2*ks + 1)
        sigma_noise = st.slider("sigma_noise",
            min_value=0.0,
            max_value=1.0,
            step=0.01)
        mean_noise = st.slider("mean_noise",
            min_value=0.0,
            max_value=1.0,
            step=0.01)

        # Setting blur levels
        res_img.blur_img(ksize, sigma_blur)
        img.blur_img(ksize, sigma_blur)
        # Setting noise levels
        res_img.noise_img(mean_noise, sigma_noise) # noise image is set. 
        img.noise_img(mean_noise, sigma_noise) 

        # Setting Target
        data = {"Blurred watermark and Noised: Kw+e": res_img.blur.image + res_img.noise.image,
        "Watermark and noise: w+e":res_img.image + res_img.noise.image,
        "watermarked: w": res_img.image,
        "image noised: x+e": img.image + img.noise.image,
        "image: x": img.image,
        "blurred image: kx": img.blur.image,
        " blurred image and noise: kx + e": img.blur.image + img.noise.image
        }
        
        selected_model = st.selectbox(
        label= "Select your model: ",
        options = data
        )
        target=data[selected_model]
        return target
    
    def run_fft_report(target, img):
        st.write("FFT running... ")
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


        def get_h_RGB(x):
            sp = lambda image: [ image[:,:,ch] for ch in range(image.shape[2]) ]
            blue , green, red = sp(x)
            zeros = np.zeros(blue.shape, np.uint8)
            blueBGR = np.stack([blue,zeros,zeros], axis=2).astype(np.uint8)
            greenBGR = np.stack([zeros,green,zeros], axis=2).astype(np.uint8)
            redBGR = np.stack([zeros,zeros,red], axis=2).astype(np.uint8)
            return np.hstack((blueBGR, greenBGR, redBGR ))

        rgb = [ target[:,:,ch] for ch in range(target.shape[2]) ] # Image with Watermark and Blur + noised
        RGB = [ img.image[:,:,ch] for ch in range(img.image.shape[2]) ] # Original Clean Image.

        h_rgb = get_h_RGB(target)
        h_RGB = get_h_RGB(img.image)


        r,g,b = rgb
        R,G,B = RGB 
        n, m = r.shape

        rgb_hat = np.zeros((n,m,3))
        for i in range(img.image.shape[2]):
            y = rgb[i]
            I = RGB[i] 
            y_af = threshold_filter(y, I, percentile=95 )
            y_hat = sp.fft.ifftn(y_af)
            rgb_hat[:,:,i] = np.abs(y_hat.reshape(n,m) )
        

        sol = (rgb_hat)
        sol = sol/np.max(sol)
        attempt = ( target )*(sol) + (1-sol)*img.image
        capt="an RGB image that shows containes the most correlation with the original image"
        Photo.show_array(255*(rgb_hat), title=capt)

        capt="RGB color channels of the original image"
        Photo.show_array( h_RGB, title=capt) 
        capt="RGB color channels of the resulting image"
        Photo.show_array( h_rgb, title=capt) 

        capt="additive model for watermark removal, X1 + X2 = X3"
        Photo.show_array( np.hstack([(1-sol)*img.image, ( target )*(sol), attempt ]), title=capt )

        capt="Each color channel of the Resulting image from removal of watermark, deblurred and denoised"
        Photo.show_array(get_h_RGB(attempt), title=capt)


        # wtr_approx =  np.abs((sol*attempt + (1-sol)*img.image) )
        wtr_approx = target/np.max(target) - np.abs((sol*attempt/np.max(attempt) + (1-sol)*img.image/np.max(img.image) ) )
        wtr_approx = 255*wtr_approx/np.max(wtr_approx)

        solution =  np.abs((sol*attempt/np.max(attempt) + (1-sol)*img.image/np.max(img.image) ) )
        solution = 255*solution/np.max(solution)

        capt="approximation of the watermark image"
        Photo.show_array(wtr_approx, capt)
        capt="Resulting image from removal of watermark, deblurred and denoised"
        Photo.show_array(solution, capt)

        st.write("percent error in image restoration percent MSE(image - solution) : ", 
            np.sum( np.square(img.image - solution))/ np.linalg.norm(img.image)**2 )
        st.write("percent error in extraction: percent  MSE(target - solution) ", 
            np.sum( np.square(target - solution))/ np.linalg.norm(target)**2 )

    def BLUE(res_img, img): 
        # Image Function
        x_best = res_img.image / 2

        # known sigma
        sigma_b = img.sigma_blur
        c = 1
        if len(res_img.image.shape) == 3:
            m, n, c = res_img.image.shape
        else:
            m, n = res_img.image.shape
        
        
        def IP_blur(x, sigma_b=sigma_b, sigma_n=img.sigma_noise, ksize=res_image.kernel_size, res_img=res_img):
            c = 1
            if len(res_img.image.shape) == 3:
                m, n, c = res_img.image.shape
            else:
                m, n = res_img.image.shape
            # setting a sigma_noise value times the identity matrix
            # finding sigma inverse squared. 
            blur = cv2.GaussianBlur(x, ksize=ksize, sigmaX=sigma_b, sigmaY=sigma_b, borderType=cv2.BORDER_CONSTANT)
            return (1/ sigma_n ) * (blur.reshape(m*n*c, ) - res_img.image.reshape(m*n*c, ))
        
        input = res_img.image.reshape(m*n*c, )
        res = least_squares(IP_blur, input, bounds=(0.0, 255.0))

        x_best = res.x.reshape(res_img.shape)

        return x_best, res.success

    # Define containers
    sidebar=st.sidebar
    container1=st.container()
    container2=st.container()
    container3=st.container()


    # sidebar is used to select image and watermark
    with sidebar:
        img_options = st.selectbox(
        label='select your base image',
        options=image_names
        )
        st.write('You selected:', img_options[:])
        img = Photo.load(fp=image_path, fname=img_options[:])
        

        selected_image = Image_Processing(fp=image_path, fname=img_options[:], img=img)

        capt_img = "Base Image is {} with size {} ".format(
            img_options[:], selected_image.shape)

        wtr_options = st.selectbox(
        label='select your watermark',
        options=watermark_names )

        st.write('You selected:', wtr_options[:])

        img = Photo.load(fp=water_path, fname=wtr_options[:])
        selected_watermark = Image_Processing(
            fp=water_path, fname=wtr_options[:], img=img)
        capt_wtr = "Watermark Image is {} with size {} ".format(
            wtr_options[:], selected_watermark.shape)
        # setting parameters
        
        scale = 0.5
        scale = st.slider("scale:",
        min_value=0.1, max_value=1.0, step=0.01)

        location = (0, 0)   
        # Adding watermark...
        h1,w1,c1 = selected_watermark.shape
        h, w, c = selected_image.shape
        sx= st.slider("sx",
        min_value = 0.0, max_value=1.0, step=0.01)
        sy= st.slider("sy",
        min_value = 0.0, max_value=1.0, step=0.01)
        location = [int(sy*(h-scale*h1)), int(sx*(w- scale*w1))]        

        alpha = 0.3 # default value
        alpha = st.slider("transparency alpha:",
        min_value=0.0, max_value=1.0, step=0.01)



    # Load target watermark...
    # Container1 is used to display the targeted image
    with container1:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(np.asarray(selected_image.image, dtype=np.uint8),
            channels="RGB", output_format="PNG", caption=capt_img)
        with col2:
            st.image(np.asarray(selected_watermark.image, dtype=np.uint8),
            channels="RGB", output_format="PNG", caption=capt_wtr)
        st.write(alpha)

        with col3:
            composite_img = selected_image.alpha_blend_watermark(selected_watermark.image, scale, location, alpha)
            w_embedded = Image_Processing(selected_image.fp, selected_image.fname+ "_" + selected_watermark.fname, composite_img)
            capt_emb = "Image {} Containing a watermark {} at location {}".format(selected_image.fname, selected_watermark.fname, location )
            st.image(np.asarray(w_embedded.image, dtype=np.uint8),
                channels="RGB", output_format="PNG", caption=capt_emb)
            res_image = Image_Processing( fp=selected_watermark.fp ,fname="Watermarked_image.jpg", img=composite_img)

    # Container2 is used to display the targeted watermark
    with container2:
        st.header("Analysis...")
        target = select_target(res_img=res_image, img=selected_image)
        run_fft_report(target, img=selected_image)
    
    with container3:
        st.header("Getting Best Linear Unbiased Estimate... BLUE Image")
        st.markdown(
            "$$ \min \| kx - y \| $$ under the constraints $ 0 \leq x \leq 255 $ "
        )

        ''' 
        # this is accross all channels and is commented out
        #    in favor of debluring the gray scaled version 
        
        r1 = Image_Processing( fp=image_path ,fname="selected_image.jpg", 
            img = selected_image.image[0:t, 0:t, :] )
        r2 = Image_Processing(fp=image_path, fname="blurred_image.jpg",
            img = target[0:t, 0:t, :] )
        '''      
        st.header( "new section: BLUE testing " )

        t = 100
        st.write("Testing with a {}x{} image... ".format(t,t))
        
        t_0, t_f = int(sx*(h-scale*h1)), int(sx*(h-scale*h1)) + t
        
        r1 = Image_Processing(fp=image_path, fname="r10by10.jpg",
            img=cv2.cvtColor(selected_image.image[ t_0: t_f,t_0: t_f, :], cv2.COLOR_RGB2GRAY) )
        
        r1.sigma_noise = selected_image.sigma_noise
        r1.sigma_blur = selected_image.sigma_blur

        r2 = Image_Processing( fp=selected_watermark.fp ,fname="target.jpg", 
            img=cv2.cvtColor(target[t_0: t_f, t_0: t_f , :], cv2.COLOR_RGB2GRAY) )

        r2.sigma_noise = selected_image.sigma_noise
        r2.sigma_blur = selected_image.sigma_blur


        st.write("loading r1 as {} and r2 as {}".format(r1.shape, r2.shape ))


        x_best, x_success = BLUE(res_img=r2, img=r1)
        import pandas as pd
        df = pd.DataFrame(x_best[:,:])
        st.write(df)



        if x_success:
            st.write("Converged to minimum \nBlue Image percent error is {}% when compared to the Original image.".format(
                np.round(100*np.sum(np.square(x_best-r1.image))/np.linalg.norm(r1.image)**2 ), 2))
        else:
            st.write("Failed to Converged to minimum \nBlue Image percent error is {}% when compared to the Original image.".format(
                np.round(100*np.sum(np.square(x_best-r1.image))/np.linalg.norm(r1.image)**2 ), 2))
        Photo.show_array(np.hstack([x_best, r2.image, r1.image]), title="BLUE Image Reconstruction")


    print("Quiting... ")

if __name__ == "__main__":
    main()