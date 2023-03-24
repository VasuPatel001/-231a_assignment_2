import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from epipolar_utils import *


'''
LLS_EIGHT_POINT_ALG  computes the fundamental matrix from matching points using 
linear least squares eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
Please see lecture notes and slides to see how the linear least squares eight
point algorithm works
'''
def lls_eight_point_alg(points1, points2):
    # form equations W.F = 0
    n = points1.shape[0]
    if points2.shape[0] != n:
        raise ValueError("Number of points mis match.")

    #Build matrix W
    W = np.zeros((n, 9))
    for i in range(n):
        W[i] = np.kron(points2[i], points1[i])

    #compute SVD of W
    Ua, Sa, V_Ta = np.linalg.svd(W)
    # Fundamental Matrix (F_hat) can be obtained as the last column of V_Ta.T or last row of V_Ta
    F_hat = V_Ta.T[:, -1].reshape((3, 3))

    #rank of F_hat is 3
    rank = np.linalg.matrix_rank(F_hat)

    #impose rank constraint of 2 by zeroing out last singular value
    U, D, V_T = np.linalg.svd(F_hat, full_matrices=True)
    F = np.dot(U, np.dot(np.diag([D[0], D[1], 0]), V_T))
    rank_f = np.linalg.matrix_rank(F)
    return F

'''
NORMALIZED_EIGHT_POINT_ALG  computes the fundamental matrix from matching points
using the normalized eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
Please see lecture notes and slides to see how the normalized eight
point algorithm works
'''
def normalized_eight_point_alg(points1, points2):
    n = points1.shape[0]
    if points2.shape[0] != n:
        raise ValueError("Number of points don't match.")

    # normalize image coordinates
    mean_1 = points1.mean(axis=0)
    x1 = np.sum(np.square(points1 - mean_1))/n
    S1 = np.sqrt(2 / x1)
    T1 = np.array([[S1, 0, -S1 * mean_1[0]],
                   [0, S1, -S1 * mean_1[1]],
                   [0, 0, 1]])
    points1 = np.dot(T1, points1.T)


    mean_2 = points2.mean(axis=0)
    x2 = np.sum(np.square(points2 - mean_2)) / n
    S2 = np.sqrt(2 / x2)
    T2 = np.array([[S2, 0, -S2 * mean_2[0]],
                   [0, S2, -S2 * mean_2[1]],
                   [0, 0, 1]])
    points2 = np.dot(T2, points2.T)

    #Compute fundamental matrix using normalized coordinates points1, x2
    F_q = lls_eight_point_alg(points1.T, points2.T)

    #De-Normalize F matrix
    F = np.dot(T2.T, np.dot(F_q, T1))
    return F
'''
PLOT_EPIPOLAR_LINES_ON_IMAGES given a pair of images and corresponding points,
draws the epipolar lines on the images
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    im1 - a HxW(xC) matrix that contains pixel values from the first image 
    im2 - a HxW(xC) matrix that contains pixel values from the second image 
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    Nothing; instead, plots the two images with the matching points and
    their corresponding epipolar lines. See Figure 1 within the problem set
    handout for an example
'''
def plot_epipolar_lines_on_images(points1, points2, im1, im2, F):

    def plot_epipolar_lines_on_image(points1, points2, im, F):
        im_height = im.shape[0]
        im_width = im.shape[1]
        lines = F.T.dot(points2.T)
        plt.imshow(im, cmap='gray')
        for line in lines.T:
            a,b,c = line
            xs = [1, im.shape[1]-1]
            ys = [(-c-a*x)/b for x in xs]
            plt.plot(xs, ys, 'r')
        for i in range(points1.shape[0]):
            x,y,_ = points1[i]
            plt.plot(x, y, '*b')
        plt.axis([0, im_width, im_height, 0])

    # We change the figsize because matplotlib has weird behavior when 
    # plotting images of different sizes next to each other. This
    # fix should be changed to something more robust.
    new_figsize = (8 * (float(max(im1.shape[1], im2.shape[1])) / min(im1.shape[1], im2.shape[1]))**2 , 6)
    fig = plt.figure(figsize=new_figsize)
    plt.subplot(121)
    plot_epipolar_lines_on_image(points1, points2, im1, F)
    plt.axis('off')
    plt.subplot(122)
    plot_epipolar_lines_on_image(points2, points1, im2, F.T)
    plt.axis('off')

'''
COMPUTE_DISTANCE_TO_EPIPOLAR_LINES  computes the average distance of a set a 
points to their corresponding epipolar lines. Compute just the average distance
from points1 to their corresponding epipolar lines (which you get from points2).
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    average_distance - the average distance of each point to the epipolar line
'''
def compute_distance_to_epipolar_lines(points1, points2, F):
    #compute epipolar line
    lines = (F.T.dot(points2.T)).T

    #compute distance to epipolar line using formula: |Ax_1 + By_1 +C | / sqrt(A^2 + B^2)
    deno = np.sqrt((lines[:, 0] ** 2) + (lines[:, 1] ** 2))
    avg_dist = np.average(np.absolute(np.sum((lines * points1), axis=1)) / deno)
    return avg_dist

if __name__ == '__main__':
    for im_set in ['data/set1', 'data/set2']:
        print('-'*80)
        print("Set:", im_set)
        print('-'*80)

        # Read in the data
        im1 = imread(im_set+'/image1.jpg')
        im2 = imread(im_set+'/image2.jpg')
        points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
        points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
        assert (points1.shape == points2.shape)

        # Running the linear least squares eight point algorithm
        F_lls = lls_eight_point_alg(points1, points2)
        print("Fundamental Matrix from LLS  8-point algorithm:\n", F_lls)
        print("Distance to lines in image 1 for LLS:", \
            compute_distance_to_epipolar_lines(points1, points2, F_lls))
        print("Distance to lines in image 2 for LLS:", \
            compute_distance_to_epipolar_lines(points2, points1, F_lls.T))


        # Running the normalized eight point algorithm
        F_normalized = normalized_eight_point_alg(points1, points2)

        pFp = [points2[i].dot(F_normalized.dot(points1[i])) 
            for i in range(points1.shape[0])]
        print("p'^T F p_normalized =", np.abs(pFp).max())
        print("Fundamental Matrix from normalized 8-point algorithm:\n", \
            F_normalized)
        print("Distance to lines in image 1 for normalized:", \
            compute_distance_to_epipolar_lines(points1, points2, F_normalized))
        print("Distance to lines in image 2 for normalized:", \
            compute_distance_to_epipolar_lines(points2, points1, F_normalized.T))

        # Plotting the epipolar lines
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_lls)
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_normalized)

        plt.show()
