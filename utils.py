# -*- coding: utf-8 -*-
'''
This is the utils file to be used by the main script suvel.py and demo.py

Written by Dr. Jiajia Liu @ University of Science and Technology of China

Revision History
2024.12.20
    Version 1.0 - Initial release

'''

import numpy as np
from matplotlib import pyplot as plt


def get_patches(input_array, nx=128, ny=128, stride=20):
    '''
    Given an input_arry, generate patches whose sizes are nx x ny
    stride is the number of pixels

    '''
    stepx = nx - stride
    stepy = ny - stride
    # Check array
    if len(input_array.shape) != 3:
        raise ValueError("The input array must have 3 dimensions!")
    if stride % 2 != 0:
        raise ValueError("Stride must be even!")

    shape = np.shape(input_array)
    # zero padding if one dimension is less than ny or nx
    dummy = np.zeros((ny if ny > shape[0] else shape[0], 
                      nx if nx > shape[1] else shape[1], shape[2]))
    dummy[0:shape[0], 0:shape[1], :] = input_array
    input_array = dummy
    shape = np.shape(input_array)

    # cut into patches if shape[0] > ny or shape[1] > nx
    if shape[0] > ny or shape[1] > nx:
        # zero padding to be the interger multiple of stepx and stepy
        xfactor = 1
        yfactor = 1
        if shape[0] > ny:
            yfactor = int(shape[0] / stepy) + 1
        if shape[1] > nx:
            xfactor = int(shape[1] / stepx) + 1
        dummy = np.zeros((yfactor * ny, xfactor * nx, shape[2]))
        dummy[0:shape[0], 0:shape[1], :] = input_array
        input_array = dummy
        shape2 = np.shape(input_array)

        rows = yfactor
        cols = xfactor
        result_list = np.zeros((rows * cols, ny, nx, shape2[2]))
        inx = 0
        for r in range(rows):
            row_start = r * stepy
            row_end = min(row_start + ny, shape2[0])
            for c in range(cols):
                col_start = c * stepx
                col_end = min(col_start + nx, shape2[1])
                sub_array = input_array[row_start:row_end, col_start:col_end, ...]
                result_list[inx, :, :, :] = sub_array
                inx = inx + 1
        return np.array(result_list)
    else:
        result_list = np.expand_dims(input_array, 0)
    
    return result_list


def merge_patches(patches, original_shape, stride=20):
    """
    Merge patches back to the original array.

    Args:
        patches: numpy array of patches obtained from get_patches function, with shape (num_patches, ny, nx, channels)
        original_shape: tuple representing the shape of the original array before patch extraction (height, width, channels)
        stride: stride value used in patch extraction (default 20)

    Returns:
        numpy array representing the merged original array
    """
    # Check array
    if len(np.shape(patches)) != 4:
        raise ValueError("The input array must have 4 dimensions!")
    if len(original_shape) != 3:
        raise ValueError("The original array must have 3 dimensions!")
    if stride % 2 != 0:
        raise ValueError("Stride must be even!")

    half_stride = int(stride / 2)

    shape = np.shape(patches)
    ny = shape[1]
    nx = shape[2]
    stepx = nx - stride
    stepy = ny - stride
    num_patches = patches.shape[0]
    # Calculate number of columns and rows of patches based on original extraction logic
    xfactor = int((original_shape[1] + stepx - 1) / stepx) if original_shape[1] > nx else 1
    yfactor = int((original_shape[0] + stepy - 1) / stepy) if original_shape[0] > ny else 1

    # Initialize an array with appropriate size for the merged result
    merged_array = np.zeros(original_shape)
    patch_index = 0
    for row_index, r in enumerate(range(yfactor)):
        row_start = r * stepy
        row_end = min(row_start + ny, original_shape[0])

        for col_index, c in enumerate(range(xfactor)):
            col_start = c * stepx
            col_end = min(col_start + nx, original_shape[1])
            if row_index == 0:
                if patch_index == 0:
                    merged_array[row_start:row_end, col_start:col_end, ...] = patches[patch_index, :row_end - row_start, :col_end - col_start, ...]
                else:
                    merged_array[row_start:row_end, col_start + half_stride:col_end, ...] = patches[patch_index, :row_end - row_start, half_stride:col_end - col_start, ...]
                    
            else:
                if col_index == 0:
                    merged_array[row_start+half_stride:row_end, col_start:col_end, ...] = patches[patch_index, half_stride:row_end - row_start, :col_end - col_start, ...]
                else:
                    merged_array[row_start + half_stride:row_end, col_start + half_stride:col_end, ...] = patches[patch_index, half_stride:row_end - row_start, half_stride:col_end - col_start, ...]
            patch_index += 1
            col_index = col_index + 1
        row_index = row_index + 1
    return merged_array
    

def plot_comparison(label, prediction, fig_num=0, ds=1, ds_unit='pixel', subscript='p',
                    vmin=-10, vmax=-10, vunit='km/s'):
    """
    Plot the comparison of data distributions between label and prediction.

    Parameters:
    label: An array containing the ground truth label data. It is expected to be a 3D array with the third dimension of size 2.
    prediction: An array containing the predicted data. Its shape must be the same as that of the label.
    fig_num: The figure number. The default value is 0.
    ds: pixel size
    ds_unit: unit for ds
    """
    # Check whether label is a 3D array and the third dimension is of size 2
    if not (isinstance(label, np.ndarray) and len(label.shape) == 3 and label.shape[2] == 2):
        raise ValueError("label must be a 3D numpy array with the third dimension of size 2")

    # Check if the shapes of prediction and label are consistent
    if label.shape!= prediction.shape:
        raise ValueError("The shapes of label and prediction must be the same")

    vx = label[:, :, 0]
    vy = label[:, :, 1]
    vxp = prediction[:, :, 0]
    vyp = prediction[:, :, 1]
    shape = np.shape(vx)

    fig = plt.figure(fig_num, figsize=(8, 8))

    # Plot the first row of subplots to show vx and vy
    ax1 = fig.add_subplot(2, 2, 1)
    im1 = ax1.imshow(vx, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower',
                     extent=(0 - shape[1]/2. * ds, shape[1]/2. * ds, 0 - shape[0]/2. * ds, shape[0]/2. * ds))
    ax1.set_xlabel('x (' + ds_unit + ')')
    ax1.set_ylabel('y (' + ds_unit + ')')
    fig.colorbar(im1, ax=ax1, fraction = 0.046, pad = 0.04)
    ax1.set_title('vx')

    ax2 = fig.add_subplot(2, 2, 2)
    im2 = ax2.imshow(vy, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower',
                     extent=(0 - shape[1]/2. * ds, shape[1]/2. * ds, 0 - shape[0]/2. * ds, shape[0]/2. * ds))

    ax2.set_xlabel('x (' + ds_unit + ')')
    ax2.set_ylabel('y (' + ds_unit + ')')
    fig.colorbar(im2, ax=ax2, fraction = 0.046, pad = 0.04)
    ax2.set_title('vy')

    # Plot the second row of subplots to show vxp and vyp
    ax3 = fig.add_subplot(2, 2, 3)
    im3 = ax3.imshow(vxp, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower',
                     extent=(0 - shape[1]/2. * ds, shape[1]/2. * ds, 0 - shape[0]/2. * ds, shape[0]/2. * ds))
    ax3.set_xlabel('x (' + ds_unit + ')')
    ax3.set_ylabel('y (' + ds_unit + ')')
    fig.colorbar(im3, ax=ax3, fraction = 0.046, pad = 0.04)
    ax3.set_title('vx_' + subscript)

    ax4 = fig.add_subplot(2, 2, 4)
    im4 = ax4.imshow(vyp, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower',
                     extent=(0 - shape[1]/2. * ds, shape[1]/2. * ds, 0 - shape[0]/2. * ds, shape[0]/2. * ds))
    ax4.set_xlabel('x (' + ds_unit + ')')
    ax4.set_ylabel('y (' + ds_unit + ')')
    fig.colorbar(im4, ax=ax4, fraction = 0.046, pad = 0.04)
    ax4.set_title('vy_' + subscript)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    print('---------------------- Test for debugging ------------------------')
    shape = (128,128, 3)
    input_array = np.random.rand(*shape)
    output_array = get_patches(input_array, stride=20)
    print(np.shape(output_array))
    reverse_array = merge_patches(output_array, shape,stride=20)
    print(np.shape(reverse_array))
    diff = reverse_array - input_array
    print(np.max(diff), np.min(diff))