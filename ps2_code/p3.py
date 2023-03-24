import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib.gridspec as gridspec
from epipolar_utils import *

'''
FACTORIZATION_METHOD The Tomasi and Kanade Factorization Method to determine
the 3D structure of the scene and the motion of the cameras.
Arguments:
    points_im1 - N points in the first image that match with points_im2
    points_im2 - N points in the second image that match with points_im1

    Both points_im1 and points_im2 are from the get_data_from_txt_file() method
Returns:
    structure - the structure matrix
    motion - the motion matrix
'''
def factorization_method(points_im1, points_im2):

    n = points_im1.shape[0]

    #step 1: compute x1_bar and x2_bar
    im1_x_bar = np.mean(points_im1, axis = 0).reshape((1,3))
    im2_x_bar = np.mean(points_im2, axis = 0).reshape((1,3))

    # compute x1_hat and x2_hat in (m, n) shape
    x1_hat = (points_im1[:,:2] - im1_x_bar[:,:2]).T
    x2_hat = (points_im2[:, :2] - im2_x_bar[:, :2]).T

    # Step 3: compute D matrix by concatenating x1_hat and x2_hat
    D = np.concatenate((x1_hat, x2_hat), axis=0)

    #Step 4: SVD of D
    U, S, V_t = np.linalg.svd(D)
    print("S:", S)
    S = np.diag([S[0], S[1], S[2]])

    # Step 5: compute structure and motion using SVD decomposition of D
    structure = S.dot(V_t[:3,:])
    motion = U[:,:3]
    return structure, motion

if __name__ == '__main__':
    for im_set in ['data/set1', 'data/set1_subset']:
        # Read in the data
        im1 = imread(im_set+'/image1.jpg')
        im2 = imread(im_set+'/image2.jpg')
        points_im1 = get_data_from_txt_file(im_set + '/pt_2D_1.txt')
        points_im2 = get_data_from_txt_file(im_set + '/pt_2D_2.txt')
        points_3d = get_data_from_txt_file(im_set + '/pt_3D.txt')
        assert (points_im1.shape == points_im2.shape)

        # Run the Factorization Method
        structure, motion = factorization_method(points_im1, points_im2)

        # Plot the structure
        fig = plt.figure()
        ax = fig.add_subplot(121, projection = '3d')
        scatter_3D_axis_equal(structure[0,:], structure[1,:], structure[2,:], ax)
        ax.set_title('Factorization Method')
        ax = fig.add_subplot(122, projection = '3d')
        scatter_3D_axis_equal(points_3d[:,0], points_3d[:,1], points_3d[:,2], ax)
        ax.set_title('Ground Truth')

        plt.show()
