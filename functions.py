import dicom

have_PIL = True
try:
    import PIL.Image
except:
    have_PIL = False

have_numpy = True
try:
    import numpy as np
except:
    have_numpy = False

def get_LUT_value(data, window, level):
    """Apply the RGB Look-Up Table for the given data and window/level value."""
    if not have_numpy:
        raise ImportError("Numpy is not available. See http://numpy.scipy.org/ to download and install")

    return np.piecewise(data,
                        [data <= (level - 0.5 - (window - 1) / 2),
                         data > (level - 0.5 + (window - 1) / 2)],
                        [0, 255, lambda data: ((data - (level - 0.5)) / (window - 1) + 0.5) * (255 - 0)])


# Display an image using the Python Imaging Library (PIL)
def createImage(dataset):
    if not have_PIL:
        raise ImportError("Python Imaging Library is not available. See http://www.pythonware.com/products/pil/ to download and install")
    if ('PixelData' not in dataset):
        raise TypeError("Cannot show image -- DICOM dataset does not have pixel data")
    
    bits = dataset.BitsAllocated
    samples = dataset.SamplesPerPixel
    if bits == 8 and samples == 1:
        mode = "L"
    elif bits == 8 and samples == 3:
        mode = "RGB"
    elif bits == 16:
        mode = "I;16"  # not sure about this -- PIL source says is 'experimental' and no documentation. Also, should bytes swap depending on endian of file and system??
    else:
        raise TypeError("Don't know PIL mode for %d BitsAllocated and %d SamplesPerPixel" % (bits, samples))

    '''if('WindowWidth' not in dataset) or ('WindowCenter' not in dataset):  # can only apply LUT if these values exist
        print("windows width and center")      
    else:
        print("no windows width and center")'''     
        
    # PIL size = (width, height)
    size = (dataset.Columns, dataset.Rows)
    #print(mode)
    #print(size)
    #print(dataset.PixelData)
    im = PIL.Image.frombuffer(mode, size, dataset.PixelData, "raw", mode, 0, 1)  # Recommended to specify all details by http://www.pythonware.com/library/pil/handbook/image.htm
    return im
