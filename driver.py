# Example 1 



# Imports

import matplotlib
from Photo import Photo as P
from Image_Processing import Image_Processing as ip
import os
from os import listdir
import numpy as np
import pandas as pd

def eda(paramaters , image_selection , results ):
    # setting random matplotlib theme for graphics 
    themes = np.array(['Solarize_Light2', '_classic_test_patch', '_mpl-gallery',
    '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background', 'fast',
    'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn', 'seaborn-bright',
    'seaborn-colorblind', 'seaborn-dark', 'seaborn-dark-palette', 'seaborn-darkgrid',
    'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel',
    'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid',
    'tableau-colorblind10'])

    fav_theme = [ themes[ind] for ind in  [20, 22, 5] ]

    #df = pd.DataFrame(themes)
    #print(df)

    #i = np.random.randint(0, len(themes))
    #style = themes[i]
    #print("Random Suggested Theme is: ", style)

    style='seaborn-talk'


    # Setting Expected key argurments
    fp = image_selection['fp']
    fname =  image_selection['fname']

    # Loading desired image X as a Photo object.
    X = P.load(fp=fp, fname=fname)
    X = ip(fp=fp, fname=fname, img=X )  # Set X as an image_processing class

    # Lets View the Photo Object and  see how we can minipulate images in python
    # X.show(title="Loaded image {} Size: {}".format(fname, X.shape ))

    # Parameters are 
    PSF_Size = paramaters["PSF_Size"]
    NS_lvl = paramaters["NS_lvl"]
    BS_lvl = paramaters['BS_lvl']

    # Setting properties of image X
    X.blur_img(ksize=PSF_Size, sigma_blur=BS_lvl)
    X.noise_img(mean_noise=0, sigma_noise=NS_lvl)

    # Plotting Blurred image X
    label = {"Shape": X.shape, "PSF_Size": PSF_Size, "Blurring Sigma":BS_lvl} 

    test_theme = fav_theme[0]
    style=style
    color_map = 'viridis'
    title = "Title: "+ fname[0:-4]
    
    save_at = results['fp']
    save_as = results['fname']

    def create_save(prop="_", save_at = results['fp'], save_as = results['fname']):
        return  os.path.join(save_at, str(prop )+"_" + save_as )
    
    img = X.blur.image + X.noise.image
    ### example 1
    # Display Photo data in different ways
    def example_1( ):
        name="Clean"
        X.show(title=name+"_"+title, save=create_save(prop=name))

        name = "Blurred"
        X.blur.show(title=name+"_"+title + "\n Blur", save=create_save(prop=name))

        
        name="Blurred_and_Noised"
        P.show_array(img, title=name+"_"+title + " Blurred and Noised\n ", save=create_save(prop=name))



        ## generating EDA for Image data
        name = "cluster_freq_map_Original"
        ip.display(X.image , title=name+"_"+title, cluster_freq=True, color_map=color_map, style=style, save=create_save(prop=name) )

        name = "cluster_freq_map_Blurred"
        ip.display(X.blur.image , title=name+"_"+title, cluster_freq=True, color_map=color_map, style=style, save=create_save(prop=name) )

        name = "cluster_freq_map_Blurred_and_Noised"
        ip.display(img , title=name+"_"+title, cluster_freq=True, color_map=color_map, style=style, save=create_save(prop=name) )





    ### example 2
    def example_2( ):
        '''
        EDA viewing Images as Signals
        '''
        ## generating EDA for Image data
        name = "cluster_amp_map_Original"
        ip.display(X.image , title=name+"_"+ title, cluster_amp=True, color_map=color_map, style=style, save=create_save(prop=name) )

        name = "cluster_amp_map_Blurred"
        ip.display(X.blur.image , title=name+"_"+ title, cluster_amp=True, color_map=color_map, style=style, save=create_save(prop=name) )

        name = "cluster_amp_map_Blurred_and_Noised"
        ip.display(img , title=name+"_"+ title, cluster_amp=True, color_map=color_map, style=style, save=create_save(prop=name) )
    
    ### example 3  
    def example_3( ):
        '''
        EDA viewing Images as Signals
        '''
        ## generating EDA for Image data
        name = "manifold_map_Original"
        ip.display(X.image , title=name+"_"+ title, manifold =True, color_map=color_map, style=style, save=create_save(prop=name) )

        name = "manifold_map_Blurred"
        ip.display(X.blur.image , title=name+"_"+ title,  manifold =True, color_map=color_map, style=style, save=create_save(prop=name) )
        
        name="manifold_map_Blurred_and_Noised"
        ip.display(img , title=name+"_"+ title,  manifold=True, color_map=color_map, style=style, save=create_save(prop=name) )

    ### example 4
    def example_4( ):
        '''
        EDA viewing Images as Signals
        and Spectro-analysis
        '''

        ## generating EDA for Image data
        name = "Spectrogram_map_Original"
        ip.display(X.image , title=name +"_"+ title, spec=True, color_map=color_map, style=style, save=create_save(prop=name) )

        name = "Spectrogram_map_Blurred"
        ip.display(X.blur.image , title=name +"_"+ title,  spec=True, color_map=color_map, style=style, save=create_save(prop=name) )
        
        name="Spectrogram_map_Blurred_and_Noised"
        ip.display(img , title=name +"_"+ title,   spec=True, color_map=color_map, style=style, save=create_save(prop=name) )

        ## generating EDA for Image data
        name = "Signals_map_Original"
        ip.display(X.image , title=name +"_"+ title, signals=True, color_map=color_map, style=style, save=create_save(prop=name) )

        name = "Signals_map_Blurred"
        ip.display(X.blur.image , title=name +"_"+ title,  signals=True, color_map=color_map, style=style, save=create_save(prop=name) )
        
        name="Signals_map_Blurred_and_Noised"
        ip.display(img , title=name +"_"+ title,   signals=True, color_map=color_map, style=style, save=create_save(prop=name) )
    
    
    
    ###
    ##
    ##
    ##
    example_1()
    example_2()
    example_3()
    example_4()






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
        'fp': os.path.join(workspace_path, "Images", "inteference_pattern"),
        'fname': fname
    }

    paramaters = { 
        "PSF_Size": (5 , 5 ),
        "NS_lvl": 0.05, 
        "BS_lvl": 11
        } 
    #eda(paramaters=paramaters, image_selection=image_selection, results=results)

    
    
    # new image and folder for EDA 
    fname = "blotchy_0039.jpg"
    image_selection={
        'fp': image_path,
        'fname': fname
        }

    results = {
        'fp': os.path.join(workspace_path, "Images", "blotchy"),
        'fname': fname
    }

    paramaters = { 
        "PSF_Size": (5 , 5 ),
        "NS_lvl": 0.05, 
        "BS_lvl": 11
        } 

    eda(paramaters=paramaters, image_selection=image_selection, results=results)











