'''data_transformations.py
Ahmed Rafique Choudhry
Performs translation, scaling, and rotation transformations on data
CS 251 / 252: Data Analysis and Visualization
Fall 2024

NOTE: All functions should be implemented from scratch using basic NumPy WITHOUT loops and high-level library calls.
'''
import numpy as np


def normalize(data):
    '''Perform min-max normalization of each variable in a dataset.

    Parameters:
    -----------
    data: ndarray. shape=(N, M). The dataset to be normalized.

    Returns:
    -----------
    ndarray. shape=(N, M). The min-max normalized dataset.
    '''
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    data_extent = data_max - data_min

    data_norm = (data - data_min) / data_extent
    return data_norm


def center(data):
    '''Center the dataset.

    Parameters:
    -----------
    data: ndarray. shape=(N, M). The dataset to be centered.

    Returns:
    -----------
    ndarray. shape=(N, M). The centered dataset.
    '''
    data_means = np.mean(data, axis=0)

    data_centered = data - data_means
    return data_centered


def rotation_matrix_3d(degrees, axis='x'):
    '''Make a 3D rotation matrix for rotating the dataset about ONE variable ("axis").

    Parameters:
    -----------
    degrees: float. Angle (in degrees) by which the dataset should be rotated.
    axis: str. Specifies the variable about which the dataset should be rotated. Assumed to be either 'x', 'y', or 'z'.

    Returns:
    -----------
    ndarray. shape=(3, 3). The 3D rotation matrix.

    NOTE: This method just CREATES and RETURNS the rotation matrix. It does NOT actually PERFORM the rotation!
    '''
    angle_rad = np.deg2rad(degrees)
    if axis == 'x':
        rotation_matrix = np.array([[1, 0, 0], [0, np.cos(angle_rad), -np.sin(angle_rad)], [0, np.sin(angle_rad), np.cos(angle_rad)]])
    if axis == 'y':
        rotation_matrix = np.array([[np.cos(angle_rad), 0, np.sin(angle_rad)], [0, 1, 0], [-np.sin(angle_rad), 0, np.cos(angle_rad)]])
    if axis == 'z':
        rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0], [np.sin(angle_rad), np.cos(angle_rad), 0], [0, 0, 1]])
    return rotation_matrix
