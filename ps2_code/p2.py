import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from p1 import *

'''
COMPUTE_EPIPOLE computes the epipole in homogenous coordinates
given matching points in two images and the fundamental matrix
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    F - the Fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    epipole - the homogenous coordinates [x y 1] of the epipole in the first image
'''
def compute_epipole(points1, points2, F):

    # Method 1: lines and epipole (e) are related by linear system of equation: [lines].[e] = 0
    # Step1: compute epipolar lines
    lines = (F.T.dot(points2.T)).T
    # Step 2: compute e using linear least square solution of lines
    U, S, V_T = np.linalg.svd(lines)
    e = V_T.T[:, -1]

    # Method 2: e is the right null vector of F
    # U, S, V_T = np.linalg.svd(F)
    # e = V_T.T[:, -1]
    return e/e[2]
    
'''
COMPUTE_MATCHING_HOMOGRAPHIES determines homographies H1 and H2 such that they
rectify a pair of images. Do not divide the homographies by their 2,2 entry.
Arguments:
    e2 - the second epipole
    F - the Fundamental matrix
    im2 - the second image
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
Returns:
    H1 - the homography associated with the first image
    H2 - the homography associated with the second image
'''
def compute_matching_homographies(e2, F, im2, points1, points2):
    #Do not divide the homographies by their 2,2 entry

    #Compute H2
    # Step 1: Compute translation matrix T
    n = points1.shape[0]
    im_height = im2.shape[0]
    im_width = im2.shape[1]
    #print(im_height, im_width)
    T = np.array([[1, 0, -im_width/2],
                  [0, 1, -im_height/2],
                  [0, 0, 1]])
    #print(T)

    #post translkation epipole
    e2_t = T.dot(e2)

    #Step 2: compute Rotation matrix
    deno = np.sqrt((e2_t[0] ** 2) + (e2_t[1] ** 2))
    alpha = -1
    if (e2_t[0] >= 0):
        alpha = 1
    a = alpha * e2_t[0] / deno
    b = alpha * e2_t[1] / deno
    R = np.array([[a, b, 0],
                  [-b, a, 0],
                  [0, 0, 1]])

    # Step 3: compute G
    # find f in (f, 0, 1) after applying rotation to e2
    f = R.dot(e2_t)
    G = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [-1/f[0], 0, 1]])

    #sanity check:
    #print("sanity check:", G.dot(R.dot(T.dot(e2))))

    # Step 3: Compute H2 = [T^-1].[G].[R].[T]
    H2 = np.linalg.inv(T).dot(G.dot(R.dot(T)))

    #Compute H1
    #Step 1: compute M = [e_cross_matrix] . F
    # e in cross matrix form
    e2_x = np.array([[0, -e2[2], e2[1]],
                     [e2[2], 0, -e2[0]],
                     [-e2[1], e2[0], 0]])
    e_vT = np.outer(e2, np.array([1, 1, 1]))
    M = (np.dot(e2_x, F) + e_vT)
    np.set_printoptions(suppress=True)

    # step 2: compute Ha
    p_hat_a = (H2 @ (M @ points1.T)).T
    p_hat_prime_a = (H2 @ points2.T).T
    # rescale p_hat
    p_hat = p_hat_a / p_hat_a[:,-1].reshape((n,1))
    p_hat_prime = p_hat_prime_a/ p_hat_prime_a[:,-1].reshape((n,1))
    # form the linear equation: Wa = b
    b = p_hat_prime[:,0].reshape((n,1))
    W = p_hat
    a = np.linalg.lstsq(W, b, rcond=None)[0]
    Ha = np.array([a[0][0], a[1][0], a[2][0],
                   0, 1, 0,
                   0, 0, 1]).reshape((3,3))
    H1 = Ha.dot(H2.dot(M))

    return H1, H2

if __name__ == '__main__':
    # Read in the data
    im_set = 'data/set1'
    im1 = imread(im_set+'/image1.jpg')
    im2 = imread(im_set+'/image2.jpg')
    points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
    points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
    assert (points1.shape == points2.shape)

    F = normalized_eight_point_alg(points1, points2)
    e1 = compute_epipole(points1, points2, F)
    e2 = compute_epipole(points2, points1, F.transpose())
    print("e1", e1)
    print("e2", e2)

    # Find the homographies needed to rectify the pair of images
    H1, H2 = compute_matching_homographies(e2, F, im2, points1, points2)
    print("H1:\n", H1)
    print('')
    print("H2:\n", H2)

    # Transforming the images by the homographies
    new_points1 = H1.dot(points1.T)
    new_points2 = H2.dot(points2.T)
    new_points1 /= new_points1[2,:]
    new_points2 /= new_points2[2,:]
    new_points1 = new_points1.T
    new_points2 = new_points2.T
    rectified_im1, offset1 = compute_rectified_image(im1, H1)
    rectified_im2, offset2 = compute_rectified_image(im2, H2)
    new_points1 -= offset1 + (0,)
    new_points2 -= offset2 + (0,)

    # Plotting the image
    F_new = normalized_eight_point_alg(new_points1, new_points2)
    plot_epipolar_lines_on_images(new_points1, new_points2, rectified_im1, rectified_im2, F_new)
    plt.show()
