import cv2
import numpy as np
from scipy.stats import entropy

import skimage.io, skimage.feature

# import mahotas
# import mahotas.demos
import numpy as np
from sklearn.cluster import KMeans
# import mahotas
# import mahotas.demos
# import mahotas as mh
import numpy as np
# from pylab import imshow, show
import matplotlib.pyplot as plt

# from new_idea import *

def sift(vid):
    video = cv2.VideoCapture(vid)
    while True:
        ret, img = video.read()
        if not ret:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        # Create SIFT object and detect keypoints
        sift = cv2.SIFT_create()
        keypoints = sift.detect(gray, None)

        # Draw keypoints on the image
        img_sift = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("", img_sift)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # save_csv("decision4.csv",header, info_4_decision)
            exit()

def otsu(img):
    # video = cv2.VideoCapture(vid)
    # while True:
    #     ret, img = video.read()
    #     if not ret:
    #         break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        s = np.identity(3, np.uint8)
        # s1 =cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)) 
        th, bina = cv2.threshold(img, 50, 200, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
        Ie = cv2.erode(bina, s)
        Id = cv2.dilate(bina, s)
        out = np.hstack((Ie, Id))
        # cv2.imshow("", out)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     # save_csv("my_data.csv",header, info_4_decision)
        #     exit()
        return out


def haralick(image):
    # loading nuclear image
    nuclear = cv2.imread(image)
    # filtering image
    nuclear = nuclear[:, :, 0]
    # adding gaussian filter
    nuclear = mahotas.gaussian_filter(nuclear, 4)
    # setting threshold
    threshed = (nuclear > nuclear.mean())
    # making is labeled image
    labeled, n = mahotas.label(threshed)
    # showing image
    imshow(labeled)
    show()
    # getting haralick features
    h_feature = mahotas.features.haralick(labeled)
    # showing the feature
    print('content  ', h_feature)
    print('len  ', h_feature.shape)
    imshow(h_feature)
    show()

import time
import cv2
import numpy as np


def scale_to_0_255(img):
    min_val = np.min(img)
    max_val = np.max(img)
    new_img = (img - min_val) / (max_val - min_val) # 0-1
    new_img *= 255
    return new_img

def apply_sliding_window_on_3_channels(img, kernel):
    # https://docs.opencv.org/4.4.0/d4/d86/group__imgproc__filter.html#ga27c049795ce870216ddfb366086b5a04
    layer_blue = cv2.filter2D(src=img[:,:,0], ddepth=-1, kernel=kernel)
    layer_green = cv2.filter2D(src=img[:,:,1], ddepth=-1, kernel=kernel)
    layer_red = cv2.filter2D(src=img[:,:,2], ddepth=-1, kernel=kernel)    
    
    new_img = np.zeros(list(layer_blue.shape) + [3])
    new_img[:,:,0], new_img[:,:,1], new_img[:,:,2] = layer_blue, layer_green, layer_red
    return new_img

def generate_gabor_bank(num_kernels, ksize=(15, 15), sigma=3, lambd=6, gamma=0.25, psi=0):
    bank = []
    theta = 0
    step = np.pi / num_kernels
    for idx in range(num_kernels):
        theta = idx * step
        # https://docs.opencv.org/4.4.0/d4/d86/group__imgproc__filter.html#gae84c92d248183bd92fa713ce51cc3599
        kernel = cv2.getGaborKernel(ksize=ksize, sigma=sigma, theta=theta, lambd=lambd, gamma=gamma, psi=psi)
        bank.append(kernel)
    return bank

def gabor(img):
    gabor_bank = generate_gabor_bank(num_kernels=4)
    
    h, w, c = img.shape
    final_out = np.zeros([h, w*(len(gabor_bank)+1), c])
    final_out[:, :w, :] = img
    
    avg_out = np.zeros(img.shape)
    
    for idx, kernel in enumerate(gabor_bank):
        res = apply_sliding_window_on_3_channels(img, kernel)
        final_out[:, (idx+1)*w:(idx+2)*w, :] = res
        kh, kw = kernel.shape[:2]
        kernel_vis = scale_to_0_255(kernel)
        final_out[:kh, (idx+1)*w:(idx+1)*w+kw, :] = np.repeat(np.expand_dims(kernel_vis, axis=2), repeats=3, axis=2)        
        avg_out += res
        
    avg_out = avg_out / len(gabor_bank)
    avg_out = avg_out.astype(np.uint8)
    return avg_out

def tradition(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r1, bw = cv2.threshold(g, 50, 200, cv2.THRESH_BINARY)
    S1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    s2 = np.ones((3,3))
    s2 = s2.astype(np.uint8)

    Ie = cv2.erode(bw, s2, iterations=2) 
    Id = cv2.dilate(bw, s2, iterations=1)
    Ib = bw - Ie
    Ig = Id - Ie
    Io = cv2.dilate(Ie, s2, iterations=3)
    # out = np.hstack((bw, Ie, Id))
    # out = np.hstack((bw, Ib, Ig, Io))
    # cv2.imshow("Result",Id)
    cv2.imshow("", Io)
    cv2.waitKey(0)

import matplotlib.pyplot as plt
import pywt
from scipy import misc
def wavelet(img):
# Load the sample image and convert it to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Compute the 2D Discrete Wavelet Transform
    # coeffs = pywt.dwt2(gray_image, 'haar')
    coeffs = pywt.dwt2(gray_image, 'haar')

    # Extract the approximation, horizontal, vertical, and diagonal coefficients
    cA, (cH, cV, cD) = coeffs

    # Plot the original and compressed images
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    ax[0,0].imshow(gray_image, cmap='gray')
    ax[0,0].set_title('Original Image')
    ax[0,1].imshow(cA, cmap='gray')
    ax[0,1].set_title('Approximation Coefficients')
    ax[1,0].imshow(cD, cmap='gray')
    ax[1,0].set_title('Horizontal Coefficients')
    ax[1,1].imshow(cV, cmap='gray')
    ax[1,1].set_title('Vertical Coefficients')
    plt.show()
    
    # cv2.imshow('', cV)
    # cv2.waitKey(0)
    return cV

def to_hsv(img):
    # Load an image
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)
    mask = cv2.inRange(hsv[:, :, 2], 0, 255)
    new_hsv = cv2.merge((h, s, mask))
    new_img = cv2.cvtColor(new_hsv, cv2.COLOR_HSV2BGR)

    return  new_img


def ignore_sunny(img):
    img = cv2.imread(img)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_value = hsv_img[:, :, 2].astype(float)
    v_min = v_value.min()
    v_max = v_value.max()
    v_value = (v_value - v_min) / (v_max - v_min) * 255
    hsv_img[:, :, 2] = v_value.astype(np.uint8)
    hist, bins = np.histogram(v_value, bins=256, range=[0, 255])
    cumulative_hist = np.cumsum(hist)
    cumulative_hist_normalized = cumulative_hist / cumulative_hist.max()
    transform = np.zeros(256)
    for i in range(256):
        transform[i] = np.floor(255 * cumulative_hist_normalized[i])
    # v_value = transform[]
    print(f'Shape = {transform.shape}')
    print(transform)
    hsv_img[:, :, 2] = v_value.astype(np.uint8)
    output_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    cv2.imshow('Result', output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def granular(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for i in range(3):
        gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)
    for i in range(3):
        gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel)
    _, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    final = img.copy()
    cv2.drawContours(final, contours, -1, (0, 255, 0), 2)

#########################
    # for cnt in contours:
    #     x,y,w,h = cv2.boundingRect(cnt)
    #     img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
    # cv2.imshow("From granular", final)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
#########################

    return final

from skimage.feature import hog
def cal_hist(img):
    # video = cv2.VideoCapture(img)
    # while True:
    #     ret, img = video.read()
    #     if not ret:
    #         break
        # img = to_hsv(img)
        # img = gabor(img)
        img = granular(img)
        # # blur = cv2.GaussianBlur(img,(5,5),0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("", img)
        # cv2.waitKey(0)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     # save_csv("decision4.csv",header, info_4_decision)
        #     exit()
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        empty = np.mean(hist[200:])
        hist_norm = hist.ravel() / hist.sum()
        entr = entropy(hist_norm)
        # print(np.argmax(hist))
        # plt.hist(img.ravel(), 256, [0, 256])
        # plt.show()
        return hist[150] / empty, entr
        # print("mean > 200: ", empty)
        # print('max gray: ',np.argmax(hist))  
        # print(hist[150]) 

   
        # cv2.imshow('', img)
        # cv2.waitKey(0)

def GLCM(img):
    # img = cv2.imread(img, 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.uint8(img*255)
    glcm = skimage.feature.graycomatrix(img, distances=[6], angles=[0], 
    levels=256, normed=True)
    dissimilarity = skimage.feature.graycoprops(P=glcm, prop='dissimilarity')
    correlation = skimage.feature.graycoprops(P=glcm, prop='correlation')
    homogeneity = skimage.feature.graycoprops(P=glcm, prop='homogeneity')
    energy = skimage.feature.graycoprops(P=glcm, prop='energy')
    contrast = skimage.feature.graycoprops(P=glcm, prop='contrast')
    ASM = skimage.feature.graycoprops(P=glcm, prop='ASM')
    return [dissimilarity, correlation, homogeneity, energy, contrast, ASM]

def canny(img):
    # img = to_hsv(img)
    edges = cv2.Canny(img, threshold1=100, threshold2=200)
    cv2.imshow("Canny Edge Detection", edges)
    cv2.waitKey(0)

# def decor(function):
#     def reverse_wrapper():
#         print(f'{function()} YOLO')
#     return reverse_wrapper


# test()
def preprocessing(img):
    # img = granular(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (7, 7), 0)
    cv2.imshow('Granular', img)
    cv2.waitKey(0)
    thresh = cv2.adaptiveThreshold(img, 255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 127, 2)
    s2 =  cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    Io = cv2.morphologyEx(thresh,cv2.MORPH_OPEN, s2, iterations=2)
    cv2.imshow('Thresh on granular', thresh)
    cv2.waitKey(0)
    return thresh

# ------------------------- AVOID BRIGHTNESS EFFECT --------------
def equalizehistogram(img):

    (b, g, r) = cv2.split(img)
    # Equalize all three channels
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    result1 = cv2.merge((bH, gH, rH))
    result1 = granular(result1)
    return result1

def gamma_correction(img):
    gamma = 0.5
    # Apply gamma correction
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)])
    gamma_corrected = cv2.LUT(img, table.astype(np.uint8))

    # Display the original and gamma corrected image
    cv2.imshow('Original Image', img)
    cv2.waitKey(0)
    cv2.imshow('Gamma Corrected Image', gamma_corrected)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def segment_kmean(img, k):
    
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    pixel_values = np.float32(img.reshape(-1, 3))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]

    segmented_image = segmented_data.reshape((img.shape))

    plt.subplot(121), plt.imshow(img), plt.title('Original Image')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(segmented_image), plt.title('Segmented Image')
    plt.xticks([]), plt.yticks([])
    plt.show()

    return cv2.cvtColor(segmented_image,cv2.COLOR_RGB2BGR)

# def cnn_partial(img):
#     def conv_(img, conv_filter):
#         filter_size = conv_filter.shape[1]
#         result = np.zeros((img.shape))
#         #Looping through the image to apply the convolution operation.
#         for r in np.uint16(np.arange(filter_size/2.0,img.shape[0]-filter_size/2.0+1)):
#             for c in np.uint16(np.arange(filter_size/2.0, img.shape[1]-filter_size/2.0+1)):
#                 curr_region = img[r-np.uint16(np.floor(filter_size/2.0)):r+np.uint16(np.ceil(filter_size/2.0)),
#                 c-np.uint16(np.floor(filter_size/2.0)):c+np.uint16(np.ceil(filter_size/2.0))]
#                 curr_result = curr_region * conv_filter
#                 conv_sum = np.sum(curr_result)
#                 result[r, c] = conv_sum 
#                 #Clipping the outliers of the result matrix.
#         final_result = result[np.uint16(filter_size/2.0):result.shape[0]-
#         np.uint16(filter_size/2.0),
#         np.uint16(filter_size/2.0):result.shape[1]-np.uint16(filter_size/2.0)]
#         return final_result

#     def conv(img, conv_filter):
#         feature_maps = np.zeros((img.shape[0]-conv_filter.shape[1]+1,
#                                     img.shape[1]-conv_filter.shape[1]+1,
#                                     conv_filter.shape[0]))
#                 # Convolving the image by the filter(s).
#         for filter_num in range(conv_filter.shape[0]):
#             curr_filter = conv_filter[filter_num, :]

#             if len(curr_filter.shape) > 2:
#                 conv_map = conv_(img[:, :, 0], curr_filter[:, :, 0])
#                 for ch_num in range(1, curr_filter.shape[-1]):
#                     conv_map = conv_map + conv_(img[:, :, ch_num], curr_filter[:, :, ch_num])

#             else: # There is just a single channel in the filter.
#                 conv_map = conv_(img, curr_filter)
#             feature_maps[:, :, filter_num] = conv_map
#             return feature_maps # Returning all feature maps.

#     def relu(feature_map):
#         #Preparing the output of the ReLU activation function.
#         relu_out = np.zeros(feature_map.shape)
#         for map_num in range(feature_map.shape[-1]):
#             for r in np.arange(0,feature_map.shape[0]):
#                 for c in np.arange(0, feature_map.shape[1]):
#                     relu_out[r, c, map_num] = np.max([feature_map[r, c, map_num], 0])
#         return relu_out
    
#     def pooling(feature_map, size=2, stride=2):
#         pool_out = np.zeros((np.uint16((feature_map.shape[0]-size+1)/stride+1), np.uint16((feature_map.shape[1]-size+1)/stride+1), feature_map.shape[-1]))
#         for map_num in range(feature_map.shape[-1]):
#             r2 = 0
#             for r in np.arange(0,feature_map.shape[0]-size+1, stride):
#                 c2 = 0
#                 for c in np.arange(0, feature_map.shape[1]-size+1, stride):
#                     pool_out[r2, c2, map_num] = np.max([feature_map[r:r+size, c:c+size]])
#                     c2 = c2 + 1
#                 r2 = r2 +1
#         return pool_out




#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     l1_filter = np.zeros((2,3,3))

#     l1_filter[0, :, :] = np.array([[[-1, 0, 1],
#                                     [-1, 0, 1],
#                                     [-1, 0, 1]]])
#     l1_filter[1, :, :] = np.array([[[1, 1, 1],
#                                     [0, 0, 0],
#                                     [-1, -1, -1]]])
#     l1_feature_map = conv(img, l1_filter)
#     l1_feature_map_relu = relu(l1_feature_map)
#     l1_feature_map_relu_pool = pooling(l1_feature_map_relu, 2, 2)

#     l2_filter = np.random.rand(3, 5, 5, l1_feature_map_relu_pool.shape[-1])
#     l2_feature_map = conv(l1_feature_map_relu_pool, l2_filter)
#     l2_feature_map_relu = relu(l2_feature_map)
#     l2_feature_map_relu_pool = pooling(l2_feature_map_relu, 2, 2)

#     l3_filter = np.random.rand(1, 7, 7, l2_feature_map_relu_pool.shape[-1])
#     l3_feature_map = conv(l2_feature_map_relu_pool, l3_filter)
#     l3_feature_map_relu = relu(l3_feature_map)
#     l3_feature_map_relu_pool = pooling(l3_feature_map_relu, 2, 2)

#     fig0, ax0 = plt.subplots(nrows=1, ncols=1)
#     ax0.imshow(img).set_cmap("gray")
#     ax0.set_title("Input Image")
#     ax0.get_xaxis().set_ticks([])
#     ax0.get_yaxis().set_ticks([])
#     plt.savefig("in_img.png", bbox_inches="tight")
#     plt.close(fig0)
#     # Layer 1
#     fig1, ax1 = plt.subplots(nrows=3, ncols=2)
#     ax1[0, 0].imshow(l1_feature_map[:, :, 0]).set_cmap("gray")
#     ax1[0, 0].get_xaxis().set_ticks([])
#     ax1[0, 0].get_yaxis().set_ticks([])
#     ax1[0, 0].set_title("L1-Map1")
#     ax1[0, 1].imshow(l1_feature_map[:, :, 1]).set_cmap("gray")
#     ax1[0, 1].get_xaxis().set_ticks([])
#     ax1[0, 1].get_yaxis().set_ticks([])
#     ax1[0, 1].set_title("L1-Map2")
#     ax1[1, 0].imshow(l1_feature_map_relu[:, :, 0]).set_cmap("gray")
#     ax1[1, 0].get_xaxis().set_ticks([])
#     ax1[1, 0].get_yaxis().set_ticks([])
#     ax1[1, 0].set_title("L1-Map1ReLU")
#     ax1[1, 1].imshow(l1_feature_map_relu[:, :, 1]).set_cmap("gray")
#     ax1[1, 1].get_xaxis().set_ticks([])
#     ax1[1, 1].get_yaxis().set_ticks([])
#     ax1[1, 1].set_title("L1-Map2ReLU")
#     ax1[2, 0].imshow(l1_feature_map_relu_pool[:, :, 0]).set_cmap("gray")
#     ax1[2, 0].get_xaxis().set_ticks([])
#     ax1[2, 0].get_yaxis().set_ticks([])
#     ax1[2, 0].set_title("L1-Map1ReLUPool")
#     ax1[2, 1].imshow(l1_feature_map_relu_pool[:, :, 1]).set_cmap("gray")
#     ax1[2, 0].get_xaxis().set_ticks([])
#     ax1[2, 0].get_yaxis().set_ticks([])
#     ax1[2, 1].set_title("L1-Map2ReLUPool")

#     # Layer 2
#     fig2, ax2 = plt.subplots(nrows=3, ncols=3)
#     ax2[0, 0].imshow(l2_feature_map[:, :, 0]).set_cmap("gray")
#     ax2[0, 0].get_xaxis().set_ticks([])
#     ax2[0, 0].get_yaxis().set_ticks([])
#     ax2[0, 0].set_title("L2-Map1")
#     ax2[0, 1].imshow(l2_feature_map[:, :, 1]).set_cmap("gray")
#     ax2[0, 1].get_xaxis().set_ticks([])
#     ax2[0, 1].get_yaxis().set_ticks([])
#     ax2[0, 1].set_title("L2-Map2")
#     ax2[0, 2].imshow(l2_feature_map[:, :, 2]).set_cmap("gray")
#     ax2[0, 2].get_xaxis().set_ticks([])
#     ax2[0, 2].get_yaxis().set_ticks([])
#     ax2[0, 2].set_title("L2-Map3")
#     ax2[1, 0].imshow(l2_feature_map_relu[:, :, 0]).set_cmap("gray")
#     ax2[1, 0].get_xaxis().set_ticks([])
#     ax2[1, 0].get_yaxis().set_ticks([])
#     ax2[1, 0].set_title("L2-Map1ReLU")
#     ax2[1, 1].imshow(l2_feature_map_relu[:, :, 1]).set_cmap("gray")
#     ax2[1, 1].get_xaxis().set_ticks([])
#     ax2[1, 1].get_yaxis().set_ticks([])
#     ax2[1, 1].set_title("L2-Map2ReLU")
#     ax2[1, 2].imshow(l2_feature_map_relu[:, :, 2]).set_cmap("gray")
#     ax2[1, 2].get_xaxis().set_ticks([])
#     ax2[1, 2].get_yaxis().set_ticks([])
#     ax2[1, 2].set_title("L2-Map3ReLU")
#     ax2[2, 0].imshow(l2_feature_map_relu_pool[:, :, 0]).set_cmap("gray")
#     ax2[2, 0].get_xaxis().set_ticks([])
#     ax2[2, 0].get_yaxis().set_ticks([])
#     ax2[2, 0].set_title("L2-Map1ReLUPool")
#     ax2[2, 1].imshow(l2_feature_map_relu_pool[:, :, 1]).set_cmap("gray")
#     ax2[2, 1].get_xaxis().set_ticks([])
#     ax2[2, 1].get_yaxis().set_ticks([])
#     ax2[2, 1].set_title("L2-Map2ReLUPool")
#     ax2[2, 2].imshow(l2_feature_map_relu_pool[:, :, 2]).set_cmap("gray")
#     ax2[2, 2].get_xaxis().set_ticks([])
#     ax2[2, 2].get_yaxis().set_ticks([])
#     ax2[2, 2].set_title("L2-Map3ReLUPool")
#     plt.savefig("L2.png", bbox_inches="tight")
#     plt.close(fig2)

#     # Layer 3
#     fig3, ax3 = plt.subplots(nrows=1, ncols=3)
#     ax3[0].imshow(l3_feature_map[:, :, 0]).set_cmap("gray")
#     ax3[0].get_xaxis().set_ticks([])
#     ax3[0].get_yaxis().set_ticks([])
#     ax3[0].set_title("L3-Map1")
#     ax3[1].imshow(l3_feature_map_relu[:, :, 0]).set_cmap("gray")
#     ax3[1].get_xaxis().set_ticks([])
#     ax3[1].get_yaxis().set_ticks([])
#     ax3[1].set_title("L3-Map1ReLU")
#     ax3[2].imshow(l3_feature_map_relu_pool[:, :, 0]).set_cmap("gray")
#     ax3[2].get_xaxis().set_ticks([])
#     ax3[2].get_yaxis().set_ticks([])
#     ax3[2].set_title("L3-Map1ReLUPool")

#     plt.show()

# a = {1: [0,4,5], 2:[4,5,6,1], 3:[5,5,6,4,5]} # 3,4 {1:3, 2:4}
# def thresh_create(a):
#     for k,v in a.items():
#         a[k] = np.sum(v)/len(v)
#     a[3] = -1
#     return a
# print(thresh_create(a))

def get_thr(arr1):
    # arr1 = np.append(arr, [27, 30])

    q1 = np.quantile(arr1, 0.25)
    q3 = np.quantile(arr1, 0.75)
    med = np.median(arr1)
    iqr = q3-q1
    upper_bound = q3+(1.5*iqr)
    lower_bound = q1-(1.5*iqr)
    # if 
    # outliers = arr1[(arr1 <= lower_bound) | (arr1 >= upper_bound)]

    # print('The following are the outliers in the boxplot:{}'.format(outliers))

    # print('Thus the array becomes{}'.format(arr1))
    plt.boxplot(arr1)
    fig = plt.figure(figsize =(10, 7))
    plt.show()
    return lower_bound, upper_bound


    
# arr = [4, 12, 15,  7, 13,  2, 12, 11, 10, 12, 15,  5,  9, 16, 17,  2, 10, 15, 4, 16, 14, 19, 12,  8, 13,  3, 16, 10,  1, 13]
# lb, ub = get_thr(arr)
# print(lb, ub)
# print(change_level(lb, ub, 27, 4))
# print(change_level(lb, ub, 10, 4))
# print(change_level(lb, ub, -5, 4))

 


