import sys
import getopt
import dicom
import PIL.Image
import PIL.ImageDraw
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import csv
import json
import glob
import os

from functions import createImage
from func1 import preprocessing, median_filt, contrast_stretch
from func2 import im_2_binary, remove_spurious_holes, biggest_gap

if len(sys.argv) == 3:

    InputFolderPath = sys.argv[1]  # "./../Annotated images/Normal/"
    OutputFolderPath = sys.argv[2]  # "./../Annotated images/Normal/out/"

    print(InputFolderPath)
    print(OutputFolderPath)

    #InputFolderPath = "/home/"+InputFolderPath
    #OutputFolderPath = "/home/"+OutputFolderPath

    
    for imgpath in glob.glob(InputFolderPath+"*.dcm"):

        inputpath = InputFolderPath + "/" + os.path.basename(imgpath)
        print("Input: ", inputpath)
        outputpath = OutputFolderPath + "/" + os.path.basename(imgpath)
        print("output:", outputpath)

        ######################################### 1. LOAD IMAGE ######################################

        plan = dicom.read_file(inputpath)
        dicom_im = createImage(plan)

        #print(plan)

        height = plan.Rows
        width = plan.Columns

        print('Rows, Columns:', height,width)
        
        #dicom_im.show()

        manufacturer = plan.Manufacturer.rstrip() # '.strip()' removes final blank spaces in strings
        valve = 'aortic'
        ##############################################################################################

        ###################### 2. OBTENTION OF THE BOUNDING BOX OF INTEREST ##########################################
        #Depending on the manufacturer (GE or Philips) the settings will be different:
        dict_settings = {'GE Vingmed Ultrasound': {'Preprocessing case': 1}, 'GE Healthcare': {'Preprocessing case': 1},
                        'GEMS Ultrasound': {'Preprocessing case': 1}, 'GE Ultrasound': {'Preprocessing case': 1},
                        'Philips Medical Systems': {'Preprocessing case': 2}}

        regions = plan.SequenceOfUltrasoundRegions # Ultrasound objects in the dicom image
        ratio = 1 
        print('Number of ultrasound objects:', len(regions))
        
        # Normally the doppler image is just 1 object, but sometimes, due to acquisition problems, 
        # the entire image appears as different unconnected objects.
        # We always look for the biggest object of 'regions' (i.e., the one with the largest area):
        numel = np.zeros(len(regions), dtype=np.int) # number of elements of each object of 'regions'
        
        for a in range(len(regions)):
            numel[a] = (regions[a].RegionLocationMaxX1 - regions[a].RegionLocationMinX0) * \
                (regions[a].RegionLocationMaxY1 - regions[a].RegionLocationMinY0)
            # With this, numel[0]: area of object 0; numel[1] = area of object 1; etc.

        item = np.argmax(numel) # largest object

        # Bounding box:
        X0 = regions[item].RegionLocationMinX0 * ratio # start column
        X1 = regions[item].RegionLocationMaxX1 * ratio # end column
        Y0 = regions[item].RegionLocationMinY0 * ratio # start row
        Y1 = regions[item].RegionLocationMaxY1 * ratio # end row

        PhysicalDeltaX = regions[item].PhysicalDeltaX / ratio
        PhysicalDeltaY = regions[item].PhysicalDeltaY / ratio

        print('--> Obtaining the bouding box...')
        im_bb = dicom_im.crop((X0-1,Y0-1,X1-1,Y1-1)) # bb: bounding box. Pillow image

        #im_bb.show() 
        ##############################################################################################################
        
        ########################### 3. PREPROCESSING IMAGE TO REMOVE ECG + OTHER INDICATORS ########################## 
        print('--> Preprocessing...')
        # We obtain the zero line (to later remove it):
        # We can find its coordinates in the metadata (and already expressed in relation 
        # to the ultrasound object):
        zero_line = regions[item].ReferencePixelY0 * ratio # row (in the ultrasound object)
        print('zero_line =', zero_line)
        '''px_im_bb = img_bb.load()
        # In GE
        print('In GE, yellow color is:', px_im_bb[0,zero_line])
        # In Philips
        #print('In Philips, brown color is:', px_im_bb[0,zero_line])'''

        preprocessing_case = dict_settings[manufacturer]['Preprocessing case']
        im_arr = preprocessing(im_bb, preprocessing_case, zero_line)
        # NOTE: 'preprocessing' returns the image array with values between 0 and 1!

        #'''plt.imshow(im_arr)
        #plt.title('Bounding box with no ECG and no marks')
        #plt.show()''' 
        ##############################################################################################################

        ########################### 4. 2D MEDIAN FILTER ########################## 
        print('--> 2D Median Filtering...')
        flow_bb_im_arr, im_arr = median_filt(im_arr,zero_line,valve)
        
        #'''plt.imshow(im_arr)
        #plt.title('2d Median Filtering')
        #plt.show()'''

        # We will present the results (the detected contour) in a plot of the Pillow image 'flow_bb_im'
        flow_bb_im = PIL.Image.fromarray((flow_bb_im_arr*255))
        #flow_bb_im.show()
        ############################################################################

        ########################### 5. CONTRAST STRETCHING ########################## 
        print('--> Contrast Stretching...')
        im_arr = contrast_stretch(im_arr)

        #'''plt.imshow(im_arr)
        #plt.title('Contrast Stretching')
        #plt.show()'''
        ##########################################################################

        ########################### 6. CONVERSION TO BINARY ##########################
        print('--> Conversion to binary...')
        # In this way, we will obtain the shape of the flow: 
        im_arr = im_2_binary(im_arr,1)

        #'''plt.imshow(im_arr)
        #plt.title('Binary image flow')
        #plt.show()'''
        ##############################################################################

        ########################### 6. REMOVE SMALL SPURIOUS AREAS ##########################
        print('--> Removing small spurious areas and filling holes...')
        im_arr_better = remove_spurious_holes(im_arr)

        #'''plt.imshow(im_arr_better)
        #plt.show()'''
        ######################################################################################

        ########################### 7. EXTRACTING THE VELOCITY PROFILE -- CONTOUR DETECTION ##########################
        print('--> Extracting the velocity profile (Biggest gap)...')
        vel_index = biggest_gap(im_arr_better,valve) # 'vel_index' is an array of ROW values of the image
        ##############################################################################################################

        ########################### 8. FILTERING THE VELOCITY PROFILE ##########################
        # We need the Heart Rate:
        if plan.HeartRate == 0: # Case of some Philips images
            HR = 60
        else:
            HR = plan.HeartRate

        print('--> Filtering the velocity profile...')
        # Lowpass first order Butterworth filter with cutoff frequency 10*HR (HR: heart rate)

        # Timeline of the flow profile:
        timeline = np.arange(0,(X1-X0+2)) * PhysicalDeltaX  
        # REMEMBER: np.arange(start,stop) generates the interval [start,stop), i.e., interval [start,stop-1]

        # Sampling frequency calculation:
        sampling_freq = (X1-X0+1)/timeline[-1]

        # Cutoff frequency definition:
        # See: http://dsp.stackexchange.com/questions/7905/converting-frequency-from-hz-to-radians-per-sample
        cutoff_freq = ((HR/60)*10 / sampling_freq) * 2

        # Lowpass first order Butterworth filter:
        order = 1
        b,a = signal.butter(order, cutoff_freq, 'low', analog=False, output='ba')

        # We use filtfilt instead of filter to keep the phase information of the signal intact
        # Otherwise, the resulting signal is delayed wrt the original
        vel_index_filtered = signal.filtfilt(b,a,vel_index)
        # vel_index_filtered is a COLUMN vector

        # After filtering, the signal amplitude is normally lower. We rescale now the signal 
        # to its original amplitude (rescaling factor equal to dividing integrals of both functions):
        rescale_factor = np.trapz(-(vel_index - vel_index[0,0])) / \
            np.trapz(-(vel_index_filtered - vel_index_filtered[0,0]))

        vel_index_filtered = vel_index_filtered * rescale_factor # ROW values
        ########################################################################################

        ########################### 9. PLOTTING THE DETECTED CONTOUR  ##########################
        print('--> Plotting the detected contour...')

        # Coordinates of the contour:
        vel_index_filtered = vel_index_filtered.reshape(vel_index_filtered.size) # rows
        x_coords_vel_index = np.arange(vel_index_filtered.size).reshape(vel_index_filtered.shape) # columns
            
        # PILLOW IMAGE 1: we draw the velocity contour in the Pillow image 'flow_bb_im' (see STEP 4):
        draw = PIL.ImageDraw.Draw(flow_bb_im)
        x_y_coords_vel_index_1 = list(zip(x_coords_vel_index,vel_index_filtered)) # with this, we obtain a list like this:
        # [(x1,y1),(x2,y2),...,(xn,yn)], which is what we need for "draw.line" (just below)
        draw.line(x_y_coords_vel_index_1, fill=1000, width=0)
        del draw

        flow_bb_im.convert('RGB')
        flow_bb_im.show()  
        # END PILLOW IMAGE 1 

        # PILLOW IMAGE 2: we draw the velocity contour in the Pillow, DICOM image 'dicom_im' (see STEP 1):
        draw = PIL.ImageDraw.Draw(dicom_im)

        if valve == 'PW_mit': # Then, the flow is ABOVE the zero_line
            x_coords_vel_index_2 = x_coords_vel_index+X0
            vel_index_filtered_2 = vel_index_filtered+Y0
            x_y_coords_vel_index_2 = list(zip(x_coords_vel_index_2,vel_index_filtered_2))
            
        else: # Then, the flow is BELOW the zero_line: we sum 'zero_line' in the columns
            x_coords_vel_index_2 = x_coords_vel_index+X0
            vel_index_filtered_2 = vel_index_filtered+Y0+zero_line
            x_y_coords_vel_index_2 = list(zip(x_coords_vel_index_2,vel_index_filtered_2))
        
        draw.line(x_y_coords_vel_index_2, fill=1000, width=0)
        del draw

        dicom_im.show()
        # END PILLOW IMAGE 2. THIS IS THE DATA THAT WE WANT TO OBTAIN FINALLY (the coords in the whole image)

        # MATPLOTLIB GRAPHICS WITH WHITE, PLAIN BACKGROUND, OF THE FLOW IN THE WHOLE DICOM IMAGE
        #plt.title('Cropped bounding box -- Detected contour');
        #plt.plot(x_coords_vel_index_2, vel_index_filtered_2, color='b',linestyle='-')
        #plt.xlim(0,width)
        #plt.ylim(height,0)

        #plt.show()
        ########################################################################################

        ######################## 10. CSV FILE CREATION ###########################
        # flow
        wr = csv.writer(open('%s_exp7_flow.csv' % outputpath,'w'), delimiter='\t')
        wr.writerows(x_y_coords_vel_index_2)

        #########################################################################

        ######################## 11. JSON FILE CREATION ###########################
        # flow

        # See http://stackoverflow.com/questions/19697846/python-csv-to-json
        # and https://www.decalage.info/en/python/print_list
        # Same as in EXPERIMENT1
        # BUT here I have the coordinates in arrays, not in lists. That's why I add "list" in the following line
        dict_data = {'x': str(list(x_coords_vel_index_2)).strip('[]'),'y': str(list(vel_index_filtered_2)).strip('[]')}

        jsonfile = open('%s_exp7_flow.json' % outputpath,'w')
        json.dump(dict_data,jsonfile)

        print('%s_exp7_flow.json' % outputpath)
        #########################################################################

else:
    print ('Execution : python exp7.py inputfilepath.dcm')