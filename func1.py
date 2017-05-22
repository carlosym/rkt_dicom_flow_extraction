import PIL.Image
import numpy as np
from scipy.signal import medfilt
from sklearn.cluster import KMeans


############################# PREPROCESSING #############################

def preprocessing(im,manufacturer,zero_line): 
    # 'im' is Pillow image 'im_bb'

    # This function:
    # 1- Removes the zero line of the image
    # 2- Averages the pixels belonging to the ECG or other undesired indicators of the image
    # with the surrounding pixels.

    width_im, height_im = im.size # columns, rows
    print('Bounding box dimensions: (width, height) =', im.size)
    
    if manufacturer == 1: # (RGB) GE images

        box = (0,zero_line-4,width_im,zero_line+4) # zero_line box

        ecg_color = (23,179,161)
        marks = (170,170,100) # yellow marks

    else: # (RGB) Philips images

        box = (0,zero_line-6,width_im,zero_line+6) # zero_line box

        ecg_color = (69,249,69)
        marks = (157,89,29) # brown marks

    # We remove the 'zero_line' from 'im':
    zero_line_region = im.crop(box) # 1st step: Crop the region of interest (box with the zero_line)
    zero_line_region = (0,0,0) # 2nd step: Change it
    im.paste(zero_line_region, box) # 3rd: Paste it in the Pillow image 'im' 

    # We obtain 'im_arr' (rows,columns,3), and we reshape it into a 3-column matrix:
    im_arr = (np.array(im).reshape(-1,3)) # (columnsÂ·rows,3)

    # Camouflage of the ECG and the marks pixels: (IS THERE A FASTER WAY?)
    for i in range(width_im*height_im):

        # if we find an ECG pixel, or a marks pixel:
        if (tuple(im_arr[i,:]) == ecg_color) or (tuple(im_arr[i,:])) == marks: 
                
            if (i-5 < 1): 
                im_arr[i,:] = np.mean(im_arr[1:i,:]) # To prevent from exceeding the image borders    
            else: 
                im_arr[i,:] = np.mean(im_arr[i-5:i,:])

    im_arr = im_arr.reshape(height_im,width_im,3)/255 # (rows,columns,3), values between 0 and 1!

    return im_arr 

#########################################################################


############################# MEDIAN FILTERING #############################
def median_filt(im_arr,zero_line,valve): 
    # 'im_arr': (rows, columns,3)

    # This function does a 2D median filtering in the image array

    # Conversion to grayscale intensity image (formula from Zolgharni:TMI:2014)
    im_arr = im_arr[:,:,0]*0.299 + im_arr[:,:,1]*0.587 + im_arr[:,:,2]*0.114
    # 'im_filt_arr' is a 2D ARRAY (rows,columns)

    # We clip the image according to the corresponding valve: above the zero line for the mitral valve
    # and below for the aortic valve.

    if valve == 'PW_mit':
        im_arr = im_arr[:zero_line+1,:] # region above the zero_line

    else:
        im_arr = im_arr[zero_line+1:,:] # region below the zero_line

    im_filt_arr = medfilt(im_arr, [3,3]) # (rows,columns)
    
    print('Filtered image dimensions: (height, width) =', im_filt_arr.shape)

    return im_arr, im_filt_arr

############################################################################

############################# CONTRAST STRETCHING #############################

def contrast_stretch(im_arr): 
    # 'im_arr': (rows, columns)

    # This function returns the contrast stretched version of the input image array

    height_im_arr, width_im_arr = im_arr.shape[0], im_arr.shape[1] # rows, columns

    # Clusterization step (2 clusters)
    kmeans = KMeans(n_clusters=2, max_iter=100).fit(im_arr.reshape(-1,1)) 
    # max_iter = 100 is the default number in kmeans matlab

    # From http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    # Attributes of the output 'kmeans':
    # cluster_centers_ : array, [n_clusters, n_features], coordinates of cluster centers
    # labels_ : Labels of each point
    # inertia_ : float, sum of distances of samples to their closest cluster center

    C = sorted(kmeans.cluster_centers_)

    # Border levels
    y1 = 0
    y2 = 1 # 1 if we have 'im_arr' has values between 0 and 1; 255 if values between 0 and 255 (I GUESS)

    imstretch = np.zeros((height_im_arr, width_im_arr))

    # Contrast stretching:

    #start = time.clock()
    for i in range(height_im_arr): # rows
        for j in range(width_im_arr): # columns

            if im_arr[i,j] >= 0 and im_arr[i,j] <= C[0]:
                imstretch[i,j] = y1

            elif im_arr[i,j] > C[0] and im_arr[i,j] <= C[1]:
                imstretch[i,j] = ( (y2-y1)/(C[1]-C[0]) )*(im_arr[i,j]-C[0]) + y1

            else:
                imstretch[i,j] = y2
    #stop = time.clock()
    #print('stop-start:', stop-start)

    #print('imstretch[39,:]:', imstretch[39,:])

    '''start = time.clock()
    
    imstretch_test = np.ones((height_im_arr, width_im_arr))*y2
    
    mask1 = (im_arr >= 0) & (im_arr <= C[0])
    mask2 = (im_arr > C[0]) & (im_arr <= C[1])
    imstretch_test[mask1] = y1
    imstretch_test[mask2] = ( (y2-y1)/(C[1]-C[0]) )*(im_arr[i,j]-C[0]) + y1
    
    stop = time.clock()
    print('Contrast stretching...')
    print('np.where(imstretch - imstretch_test != 0) =', np.where(imstretch - imstretch_test != 0))
    print('stop-start:', stop-start)'''

    # IT DOES NOT WORK! (DIFFERENT RESULTS) WHYYYYYYY?????
    #(I HOPE I CAN CORRECT IT, BECAUSE IT IS FASTER THAN THE FOR LOOP)

    return imstretch

###############################################################################
   