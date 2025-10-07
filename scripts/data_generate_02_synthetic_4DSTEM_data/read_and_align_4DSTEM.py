import h5py
import numpy as np
from skimage import transform
import sys

def read_4D(fname, trim_meta = True):

    '''
    Read the 4D dataset as a numpy array from .raw , . mat, .npy file.
    Input:

    fname: the file path

    Return: 

    dp       : numpy array
    dp_shape : the shape of the data

    '''

    fname_end = fname.split('.')[-1]

    if fname_end == 'raw':
        with open(fname, 'rb') as file:
            dp = np.fromfile(file, np.float32)

        columns = 128    
        rows = 130

        # print("dp.shape", dp.shape)
            
        sqpix = dp.size/columns/rows
        #Assuming square scan, i.e. same number of x and y scan points
        pix = int(sqpix**(0.5))
        
        dp = np.reshape(dp, (pix, pix, 130, 128), order = 'C')
        
        # Trim off the last two meta data rows if desired.  
        # The meta data is for EMPAD debugging, 
        # and generally doesn't need to be kept.
        if trim_meta:
            # dp = dp[:,:,0:128,:]
            # print("dp[:,:,128:,:].shape", dp[:,:,128:,:].shape, "\n")
            # print("dp[:,:,128:,:]\n", dp[:,:,128:,:], "\n")
            # print("dp[:,:,128,:]\n", dp[:,:,128,:], "\n")
            # print("dp[:,:,129,:]\n", dp[:,:,129,:], "\n")
            dp = dp[:,:,:128,:]

    ## Read 4D data from .mat file

    elif fname_end == 'mat':

        with h5py.File(fname, "r") as f:
            
            data_name = list(f.keys())[0]
            dp = np.array(list(f[data_name]))
    elif fname_end == 'npy':
        dp = np.load(fname)
    else:
        print('The Format is WRONG!! Only support .mat , .raw & .npy file !!') 


    sel = dp < 1
    dp[sel] = 1
    
    return dp

def alignment(cbed_data):

    '''

    Align the diffraction patterns through the Center of mass of the center beam
    '''

    x, y, kx, ky = np.shape(cbed_data)
    com_x, com_y = quickCOM(cbed_data) # need to add
    cbed_tran    = np.zeros((x, y, kx, ky))
    
    for i in range(x):
        for j in range(y):
            afine_tf = transform.AffineTransform(translation=(-kx//2+com_x[i,j], -ky//2+com_y[i,j]))
            cbed_tran[i,j,:,:] = transform.warp(cbed_data[i,j,:,:], inverse_map=afine_tf)
        sys.stdout.write('\r %d,%d' % (i, j) + ' '*10)
    com_x2, com_y2 = quickCOM(cbed_tran)
    std_com = (np.std(com_x2), np.std(com_y2))
    mean_com = (np.mean(com_x2), np.mean(com_y2))
    
    return cbed_tran, mean_com, std_com

def quickCOM(cbed_data):
    x, y, kx, ky = np.shape(cbed_data)
    center_x = kx//2 ; center_y = ky//2 
    disk = 5
    mask = spotmask(center_x,center_y, kx, disk)
    
    ap2_x, ap2_y = centroid2(cbed_data,x, y, kx, mask)
    
    return ap2_x, ap2_y

def spotmask(center_x,center_y, kx, disk):
    
    innerDisk   = disk

    mask = np.zeros((kx,kx))
    for i in range(kx):
        for j in range(kx):
            if (i - center_x) ** 2 + (j - center_y) ** 2 < innerDisk ** 2:
                mask[i][j] = 1

    return mask

def centroid2(fun, x, y, kx, mask):

    ap2_x = np.zeros((x,y)); ap2_y = np.zeros((x,y))
    rx, ry  = np.meshgrid(kx, kx)
    vx = np.arange(kx); vy = np.arange(kx)
    for i in range(x):
        for j in range(y):
            cbed = np.squeeze(fun[i,j, :, :] * mask)
            pnorm = np.sum(cbed)
            ap2_x[i,j] = np.sum(vx * np.sum(cbed, axis = 0))/pnorm
            ap2_y[i,j] = np.sum(vy * np.sum(cbed, axis = 1))/pnorm
            
    return ap2_x, ap2_y
