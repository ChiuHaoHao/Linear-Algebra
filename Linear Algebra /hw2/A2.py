# 2019 Linear Algebra Assignment 2
# import the necessary packages
import numpy as np
import matplotlib.image as mtimg
import matplotlib.pyplot as mtplot

def get_image_size(src):
    ### give the corner points of the object in the image 
    ### return the points of the scanned image

    (tl, tr, br, bl) = src
 
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
 
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]])
    
    return dst

    
def four_point_transform(src, dst):  
    # compute the perspective transform matrix and then apply it
    # the transform is found by solving ax = b
    
    # for perspective projection, the 3x3 matrix has 8 unknows
    #      [t11 t12 t13] [s1]   [v1]
    #  T = [t21 t22 t23] [s2] = [v2]
    #      [t31 t32  1 ] [ 1]   [v3]
    #
    #  where d1 = v1/v3 and d2 = v2/v3.
    #  
    #  t11*s1 + t12*s2 + t13 = v1 = d1*v3 = d1*(t31*s1+t32*s2+1)
    #  t21*s1 + t22*s2 + t23 = v2 = d2*v3 = d2*(t31*s1+t32*s2+1)
    #  t31*s1 + t32*s2 + 1   = v3
    
    A = np.zeros((8, 8))
    b = np.zeros((8, 1))

    # assigning values
    # ******* your code here ********
    A = np.array([
        [src[0][0], src[0][1], 1,          0,         0, 0, 0, 0], 
        [        0,         0, 0,  src[0][0], src[0][1], 1, 0, 0], 
        [src[1][0], src[1][1], 1,          0,         0, 0, dst[1][0]*(src[1][0])*(-1), dst[1][0]*(src[1][1])*(-1)], 
        [        0,         0, 0,  src[1][0], src[1][1], 1, 0, 0], 
        [src[2][0], src[2][1], 1,          0,         0, 0, dst[1][0]*(src[2][0])*(-1), dst[1][0]*(src[2][1])*(-1)],
        [        0,         0, 0,  src[2][0], src[2][1], 1, dst[2][1]*(src[2][0])*(-1), dst[2][1]*(src[2][1])*(-1)], 
        [src[3][0], src[3][1], 1,          0,         0, 0, 0, 0],
        [        0,         0, 0,  src[3][0], src[3][1], 1, dst[2][1]*(src[3][0])*(-1), dst[2][1]*(src[3][1])*(-1)]  
        ])

    b[0][0] = 0
    b[1][0] = 0
    b[2][0] = dst[1][0]
    b[3][0] = 0
    b[4][0] = dst[1][0]
    b[5][0] = dst[2][1]
    b[6][0] = 0
    b[7][0] = dst[2][1]
    

    # now put the solution x into a 3x3 matrix T
    # the t33 element = 1
    x = np.linalg.solve(A, b)
    x = np.concatenate((x, [[1]]), axis=0)
    T = x.reshape(3,3)
     
    # We need the inverse transformation, so R = np.inv(T)
    R = np.linalg.inv(T)

    # the r33 should be 1, so normalize it. 
    R = R/R[2][2]

    print (R)
    
    return R

def perspective_projection(size, image, R):
    ### Convert the original 3D image to 2D
    # img is the dst picture
    img = np.zeros((size[0], size[1], 3))
    [w, h, c] = np.array(image).shape
    v = np.zeros((3,1))
    v[2] = 1
    x = np.zeros((3,1))
    # transformation is done by finding the 'donor' of points in dst picture
    # *********** your code here *************
    for i in range (size[0]):
        for j in range (size[1]):
            v[0] = j
            v[1] = i
            x = np.dot(R,v)
            b = int (x[0][0]/x[2][0])
            a = int (x[1][0]/x[2][0])
            img[i][j] = image[a][b]
    
    warped = img.astype(np.uint8)

    # return the warped image
    return warped

def interpolate_transform(src, size, image):
    (tl, tr, br, bl) = src
    [w, h, c] = np.array(image).shape
    img = np.zeros((size[0], size[1], 3))
    
    # ********** your code here ******************
    for i in range (size[0]): #i==alpha
        for j in range (size[1]): #j==beta
            tmp1 = j/size[1]
            tmp2 = i/size[0]
            y =int ((1-tmp1)*(1-tmp2)*(tl[0]) + (tmp1)*(1-tmp2)*(tr[0]) + (tmp2)*(1-tmp1)*(bl[0]) + (tmp1)*(tmp2)*(br[0]))
            x =int ((1-tmp1)*(1-tmp2)*(tl[1]) + (tmp1)*(1-tmp2)*(tr[1]) + (tmp2)*(1-tmp1)*(bl[1]) + (tmp1)*(tmp2)*(br[1]))
            img[i][j] = image[x][y]

    warped = img.astype(np.uint8)

    # return the warped image
    return warped

# -------- main program -------------------------
image = mtimg.imread('hw2_1.jpg')
src = np.array([[29,181], [519,67], [951,667], [293,917]])
#(29,181),(519,67),(951,667),(293,917)
dst = get_image_size(src)
R   = four_point_transform(src, dst)

img1 = perspective_projection([dst[2][1]+1, dst[2][0]+1], image, R)
img2 = interpolate_transform(src, [dst[2][1]+1, dst[2][0]+1], image)

mtplot.subplot(4, 3, 1)
mtplot.imshow(image)
mtplot.subplot(4, 3, 2)
mtplot.imshow(img1)
mtplot.subplot(4, 3, 3)
mtplot.imshow(img2)

# image1 = mtimg.imread('hw2_2.jpg')
# src1 = np.array([[29,67], [535,51], [965,621], [127,759]])
# #(239,331),(737,107),(857,364),(203,648)
# dst1 = get_image_size(src1)
# R1   = four_point_transform(src1, dst1)

# img3 = perspective_projection([dst1[2][1]+1, dst1[2][0]+1], image1, R1)
# img4 = interpolate_transform(src1, [dst1[2][1]+1, dst1[2][0]+1], image1)

# mtplot.subplot(4, 3, 4)
# mtplot.imshow(image1)
# mtplot.subplot(4, 3, 5)
# mtplot.imshow(img3)
# mtplot.subplot(4, 3, 6)
# mtplot.imshow(img4)

# image2 = mtimg.imread('hw2_3.jpg')
# src2 = np.array([[197,74], [501,280], [311,529], [4,280]])
# #(239,331),(737,107),(857,364),(203,648)
# dst2 = get_image_size(src2)
# R2   = four_point_transform(src2, dst2)

# img5 = perspective_projection([dst2[2][1]+1, dst2[2][0]+1], image2, R2)
# img6 = interpolate_transform(src2, [dst2[2][1]+1, dst2[2][0]+1], image2)

# mtplot.subplot(4, 3, 7)
# mtplot.imshow(image2)
# mtplot.subplot(4, 3, 8)
# mtplot.imshow(img5)
# mtplot.subplot(4, 3, 9)
# mtplot.imshow(img6)

# image3 = mtimg.imread('hw2_4.jpg')
# src3 = np.array([[321,125], [915,213], [763,545], [7,327]])
# #(239,331),(737,107),(857,364),(203,648)
# dst3 = get_image_size(src3)
# R3   = four_point_transform(src3, dst3)

# img7 = perspective_projection([dst3[2][1]+1, dst3[2][0]+1], image3, R3)
# img8 = interpolate_transform(src3, [dst3[2][1]+1, dst3[2][0]+1], image3)

# mtplot.subplot(4, 3, 10)
# mtplot.imshow(image3)
# mtplot.subplot(4, 3, 11)
# mtplot.imshow(img7)
# mtplot.subplot(4, 3, 12)
# mtplot.imshow(img8)