from numpy import zeros, array, clip, trunc
from math import pi, cos, sqrt
from cv2 import imread, imwrite, imshow, waitKey, destroyAllWindows, cvtColor, COLOR_BGR2RGB
import sys
import matplotlib.pyplot as plt
cos_backup = array([])
def cos_values(N):
    ret = zeros((N,N))
    for n in range(len(ret)):
        for k in range(len(ret[n])):
            ret[k,n] = cos(((pi*k)*(2*n+1))/(2*N))
    global cos_backup
    cos_backup = ret
def direct_dct(vector):
    N = len(vector)
    if len(cos_backup) != N:
        cos_values(N)
    vector = cos_backup.dot(vector)
    vector[0] = vector[0] * sqrt(1/2)
    vector = vector * sqrt(2/N)
    return vector
def inverse_dct(vector):
    N = len(vector)
    if len(cos_backup) != N:
        cos_values(N)
    vector[0] = vector[0] * sqrt(1/2)
    vector = vector * sqrt(2/N)
    return cos_backup.T.dot(vector)
def direct_dct_2d(matrix):
    Nx,Ny = matrix.shape
    for line in range(Nx):
        matrix[line] = direct_dct(matrix[line])
    for column in range(Ny):
        matrix[:,column] = direct_dct(matrix[:,column])
    return matrix
def inverse_dct_2d(matrix):
    Nx,Ny = matrix.shape
    for column in range(Ny):
        matrix[:,column] = inverse_dct(matrix[:,column])
    for line in range(Nx):
        matrix[line] = inverse_dct(matrix[line])
    return matrix
def direct_dct_image(img):
    if img.shape[2] == 3:
        for i in range(3):
            img[:,:,i] = direct_dct_2d(img[:,:,i])
    else:
        img[:, :, 0] = direct_dct_2d(img[:, :,0])
    return img
def inverse_dct_image(img):
    if img.shape[2] == 3:
        for i in range(3):
            img[:,:,i] = inverse_dct_2d(img[:,:,i])
    else:
        img[:, :, 0] = inverse_dct_2d(img[:, :,0])
    return img.clip(0, 255)
def remove_coeficients_from_image(img, keep):
    img[keep:, :, :] = 0
    img[:, keep:, :] = 0
    return img


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python3 dctopt.py [IMAGE FILE] [NUMBER OF COEFICIENTS TO KEEP]')
        quit()

    rows = 1
    columns = 4

    fig = plt.figure()

    img = imread(sys.argv[1])
    img = img.astype('float64')
    fig.add_subplot(rows,columns,1)
    plt.imshow(cvtColor(img.astype('uint8'), COLOR_BGR2RGB))
    fig.add_subplot(rows, columns, 2)
    x = direct_dct_image(img.copy())
    plt.imshow(cvtColor(x.astype('uint8'), COLOR_BGR2RGB))
    fig.add_subplot(rows, columns, 3)
    y = remove_coeficients_from_image(x.copy(), int(sys.argv[2]))
    plt.imshow(cvtColor(y.astype('uint8'), COLOR_BGR2RGB))
    fig.add_subplot(rows, columns, 4)
    plt.imshow(cvtColor(inverse_dct_image(y).astype('uint8'), COLOR_BGR2RGB))
    
    plt.show()

    #TEST 1
    # rows = 1
    # columns = 3
    # vector_test = array([8,16,24,32,40,48,56,64])
    # x = direct_dct(vector_test)
    # fig = plt.figure()
    # fig.add_subplot(rows, columns, 1)
    # plt.plot(vector_test)
    # fig.add_subplot(rows, columns, 2)
    # plt.plot(x)
    # fig.add_subplot(rows, columns, 3)
    # plt.plot(inverse_dct(x))
    # plt.show()



    # TEST 2
    # matrix_test = array([139,144,149,153,155,155,155,155,
    #                    144,151,153,156,159,156,156,156,
    #                    150,155,160,163,158,156,156,156,
    #                    159,161,162,160,160,159,159,159,
    #                    159,160,161,162,162,155,155,155,
    #                    161,161,161,161,160,157,157,157,
    #                    162,162,161,163,162,157,157,157,
    #                    162,162,161,161,163,158,158,158],dtype='float64').reshape((8,8))
    # print(matrix_test)
    # rows = 1
    # columns = 3
    # fig = plt.figure()
    # fig.add_subplot(rows,columns,1)
    # plt.imshow(matrix_test)
    # x = direct_dct_2d(matrix_test.copy())
    # print(x)
    # fig.add_subplot(rows,columns,2)
    # plt.imshow(x)
    # y = inverse_dct_2d(x)
    # print(y)
    # fig.add_subplot(rows, columns, 3)
    # plt.imshow(y)
    #
    # plt.show()

    # rows = 1
    # columns = 3
    #
    # fig = plt.figure()
    #
    # img = imread('lena256color.jpg')
    # img = img.astype('float64')
    # fig.add_subplot(rows,columns,1)
    # plt.imshow(cvtColor(img.astype('uint8'), COLOR_BGR2RGB))
    # fig.add_subplot(rows, columns, 2)
    # x = direct_dct_image(img.copy())
    # plt.imshow(cvtColor(x.astype('uint8'), COLOR_BGR2RGB))
    # fig.add_subplot(rows, columns, 3)
    # plt.imshow(cvtColor(inverse_dct_image(x).astype('uint8'), COLOR_BGR2RGB))
    #
    # plt.show()
    pass
