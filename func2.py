import PIL.Image
import numpy as np
from skimage.measure import label, regionprops
from skimage.filter import threshold_otsu
from scipy.ndimage.morphology import binary_fill_holes


############################# BINARIZATION #############################

def im_2_binary(im_arr, Mode):
	# 'im_arr': (rows, columns) 

	# Inputs
	# im_arr: imput image array that will be converted to binary format
	# Mode: Mode = 1 -> threshold as in Zolgharni et al.; Mode = 2 -> Otsu's threhold

	# Conversion to grayscale intensity image (formula from Zolgharni:TMI:2014)

	if Mode == 1:

		threshold = np.linspace(0.01,1,100, endpoint=True)
		count_old = 0
		change = np.zeros(len(threshold))

		# The optimum threshold to binarize the image is considered as the value
	    # for which the largest number of pixels turn from 1 to 0.

		for i in range(len(threshold)):
			
			count_new = np.sum(im_arr < threshold[i])
			change[i] = count_new - count_old
			count_old = count_new

		cutoff_ind = np.argmax(change)
		cutoff = threshold[cutoff_ind]
		#print('cutoff =', cutoff)

		im_arr_bin = (im_arr > cutoff*15)*1 # 'bool_array*1' transforms a boolean array into an int array

	elif Mode == 2:

		# Implementation of the Otsu's method --> clustering based image
	    # thresholding. Minimal intra-class variance, maximal inter-class
	    # variance.

	    thresh = threshold_otsu(im_arr)
	    #print('threshold =', thresh)
	    im_arr_bin = (im_arr > thresh)*1 # 'bool_array*1' trasnforms a boolean array into an int array

	return im_arr_bin

########################################################################

################### REMOVING SPURIOUS AREAS AND HOLES ###################

def remove_spurious_holes(im_arr):
	# 'im_arr': (rows, columns)

	# This function:
	# 1- sets to zero the clusters with less than 'threshold' connected pixels
	# 2- fills the holes of the binary image

	print('	Removing spurious areas...')
	# Evaluation of the connected components in the image:
	label_im = label(im_arr)
	props = regionprops(label_im)

	# See the properties of the output 'props' here: http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops
	# area : int, Number of pixels of region.
	# coords : (N, 2) ndarray, coordinate list (row, col) of the region.

	threshold = 500 # Empirical value found in Zolgharni_TMI

	# Count the number of elements in each connected component:
	numPixels = [region.area for region in props]

	# Find those clusters with less than 'threshold' pixels:
	idx = np.where(np.array(numPixels) < threshold)[0]
	
	# Set clusters to 0:
	im_new = np.copy(im_arr)
	coords =  []

	for i in range(len(idx)):
		conn_object_coords = props[idx[i]].coords
					
		for j in range(len(conn_object_coords)):
			coords = tuple(conn_object_coords[j])
			im_new[coords] = 0

	print('	Filling holes...')
	# See: https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.morphology.binary_fill_holes.html
	im_new = binary_fill_holes(im_new)

	return im_new

#########################################################################

############################# BIGGEST GAP #############################

def biggest_gap(im_arr, valve):
	# 'im_arr': (rows, columns)

	# FROM SERGIO GÃ“MEZ's CODE (IN MATLAB):
	# From Zolgharni_TMI: 
	# The maximum velocity profile is then extracted from the resulting filtered
	# image by using the biggest-gap method [4]. This is done by sweeping the image
	# from left to right. Each column of the image represents a vector containing 
	# black and white pixels. The gap is defined as a cluster of consecutive black
	# pixels, and the pixel at the beginning of the largest gap from top is selected
	# as one point on the velocity profile. This method allows isolating the desired
	# Doppler envelope from the aliased signal.

	# I CHANGED IT A LITTLE BIT BECAUSE THE BACKGROUND IS NOT ALWAYS THE BIGGEST GAP
	# SEE ***MY CHANGES*** BELOW
	
	height_im_arr, width_im_arr = im_arr.shape[0], im_arr.shape[1] # rows, columns
	index = np.zeros((1, width_im_arr))

	if valve == 'PW_mit':
	    im_arr = np.flipud(im_arr)

	im_arr_bool = np.array(im_arr, dtype=bool)

	for i in range(width_im_arr):
		
		col_i = ~im_arr_bool[:,i]*1 # 'bool_array*1' transforms a boolean array into an int array
		copy_col_i = np.copy(col_i)

		A = np.column_stack((col_i, copy_col_i))
		label_A = label(A)
		props = regionprops(label_A)
		# See the properties of the output 'props' here: http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops

		# Count the number of elements in each connected component:
		numPixels = [region.area for region in props]
		
		if numPixels: # if numPixels is NOT empty

			# Find the biggest cluster:
			idx = np.argmax(numPixels)

			# We only will look at the results of the fisrt column of A:
			all_coords = props[idx].coords[:,0]
			
			# ***MY CHANGES (START)***
			values_at_image = im_arr[all_coords[0],i]
			if values_at_image == 1: # the (found) largest region is a FLOW REGION, so ->
				# -> the INDEX that we are looking for is the LAST one from the (found) region.

				idx = all_coords[-1]

			else:  # the (found) largest region is a NON-FLOW REGION, so ->
				# -> the INDEX that we are looking for is the FIRST one from the (found) region.

				idx = all_coords[0] 
			
			# ***MY CHANGES (END)***

		else: #  If there is not any single hole, we preserve the last index value
			idx = index[0,i-1]
		
		index[0,i] = idx

	return index

#######################################################################