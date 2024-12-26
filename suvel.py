# -*- coding: utf-8 -*-
'''
This is the main script for SUVEL

Written by Dr. Jiajia Liu @ University of Science and Technology of China

Cite Liu et al. 2025, Tracking the Photospheric Horizontal Velocity Field with Shallow U-Net Models

Revision History
2024.12.20
    Version 1.0 - Initial release

'''

import numpy as np
import tensorflow as tf
from utils import get_patches, merge_patches


def custom_loss():
    '''
    This is a custom loss function used in the models
    '''
    def loss(y_true, y_pred):
        # Extract vx and vy from both the true and predicted values
        vx_true = y_true[..., 0]
        vy_true = y_true[..., 1]
        vx_pred = y_pred[..., 0]
        vy_pred = y_pred[..., 1]
        
        # L2: MAE of vx
        L2 = tf.reduce_mean(tf.abs(vx_true - vx_pred))
        #MSE
        L2s = tf.reduce_mean(tf.square(vx_true - vx_pred)) # mean square error
        # L3: MAE of vy
        L3 = tf.reduce_mean(tf.abs(vy_true - vy_pred))
        #MSE
        L3s = tf.reduce_mean(tf.square(vy_true - vy_pred)) # mean square error
        total_loss = L2 + L3 + 10* (L2s + L3s) # because mae is 10 times of mse in this particular case
        
        return tf.reduce_mean(total_loss)
    
    return loss


def process_data(intensity=None, magnetic=None):
    '''
    Parameters
    ----------
    intensity : numpy array, optional
        Photospheric intensity. The default is None. If it is not None, 
        it has to be ny x nx x 3 and normalized to [0, 1].
        The default is None.
    magnetic : numpy array, optional
        Photospheric vertical or LOS magnetic field strength. The default is None.
        If it is not None, it has to be ny x nx x 3 and normalized to [0, 1].

    Returns
    -------
    intensity_patches
        patches of the intensity with size of 128 x 128
    magnetic_patches
        patches of the magnetic field strength with size of 128 x 128

    '''
    
    if intensity is None and magnetic is None:
        print("Error: Both intensity and magnetic cannot be None.")
        raise ValueError("Both inputs cannot be None")


    if (intensity is not None and len(intensity.shape)!= 3) or (magnetic is not None and len(magnetic.shape)!= 3):
        print("Error: Input arrays must be three-dimensional.")
        raise ValueError("Input arrays must be three-dimensional")


    if (intensity is not None and intensity.shape[2]!= 3) or (magnetic is not None and magnetic.shape[2]!= 3):
        print("Error: The third dimension of the input arrays must be 3.")
        raise ValueError("The third dimension of input arrays must be 3")


    def check_range(arr):
        return np.all((arr >= 0) & (arr <= 1))

    if intensity is not None and not check_range(intensity):
        print("Error: Intensity array values must be in the range [0, 1].")
        raise ValueError("Intensity array values out of range")
    if magnetic is not None and not check_range(magnetic):
        print("Error: Magnetic array values must be in the range [0, 1].")
        raise ValueError("Magnetic array values out of range")
    
    intensity_patches = None
    magnetic_patches = None
    
    if intensity is not None:
        intensity_patches = get_patches(intensity, nx=128, ny=128, stride=20)
    if magnetic is not None:
        magnetic_patches = get_patches(magnetic, nx=128, ny=128, stride=20)

    return intensity_patches, magnetic_patches


def suvel(intensity=None, magnetic=None, model_path='./'):
    '''
    Parameters
    ----------
    intensity : numpy array, optional
        Photospheric intensity. The default is None. If it is not None, 
        it has to be ny x nx x 3 and normalized to [0, 1].
        The default is None.
    magnetic : numpy array, optional
        Photospheric vertical or LOS magnetic field strength. The default is None.
        If it is not None, it has to be ny x nx x 3 and normalized to [0, 1].
    model_path : string, optional
        Path that store the SUVEL models. The default is './'.

    Returns
    -------
    result : numpy array
        The predicted velocity field, with a size of ny x nx x 2.

    '''
    if intensity is None and magnetic is None:
        raise ValueError("Nothing has been passed!")

    # check data and get patches
    intensity_patches, magnetic_patches = process_data(intensity, magnetic)
    
    # read and load models
    intensity_model = tf.keras.models.load_model(model_path + 'intensity_model.h5', custom_objects={'loss': custom_loss()})
    magnetic_model = tf.keras.models.load_model(model_path + 'magnetic_model.h5', custom_objects={'loss': custom_loss()})
    hybrid_model = tf.keras.models.load_model(model_path + 'hybrid_model.h5', custom_objects={'loss': custom_loss()})
    
    # use the intensity model
    if intensity is not None:
        shape = np.shape(intensity)
        result_i = intensity_model.predict(intensity_patches)

    # use the magnetic model
    if magnetic is not None:
        shape = np.shape(magnetic)
        result_m = magnetic_model.predict(magnetic_patches)
    
    # use the hybrid model
    if intensity is not None and magnetic is not None:
        input_h = np.concatenate((intensity_patches, magnetic_patches, result_i, result_m), axis=3)
        print(np.shape(input_h))
        result_h = hybrid_model.predict(input_h)
        result = merge_patches(result_h, (shape[0], shape[1], 2))
        return result
    else:
        if intensity is not None:
            result = merge_patches(result_i, (shape[0], shape[1], 2))
            return result
        else:
            result = merge_patches(result_m, (shape[0], shape[1], 2))
            return result



if __name__ == '__main__':
    pass