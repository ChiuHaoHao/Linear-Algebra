
# 2019 Linear Algebra assignment 1 example

import numpy as np

# create the cipher matrix
A = np.array(
    [[0, 1, 0, 2, 2],
     [6, 0, 1, 0, 0],
     [0, 0, 0, 1, 0],
     [7, 0, 0, 0, 1],
     [8, 0, 0, 0, 1]])

np.linalg.det(A)

# x is the original message
x1 = np.array(
    [[1],
     [2],
     [3],
     [4],
     [5]])

# y is the encoded message
y1 = np.dot(A, x1)
print(y1)

# define a function to compute inverse
#------------------------------------------------
def creat_diagnal_matrix(number):
    matrix=np.zeros((number,number),dtype=float)    
    for i in range(number):
        matrix[i][i]=1
    return matrix

def my_inverse_matrix(a):
    diagnoal=creat_diagnal_matrix(5)
    for i in range(5):
        a[3][i] += a[4][i]*(-1) 
        diagnoal[3][i] += diagnoal[4][i]*(-1)

    for i in range(5):
        a[4][i] += a[3][i]*8 
        diagnoal[4][i] += diagnoal[3][i]*8
    for i in range(5):
        a[0][i] += a[3][i]*(-1) 
        diagnoal[0][i] += diagnoal[3][i]*(-1)

    for i in range(5):
        a[1][i] += a[3][i]*6 
        diagnoal[1][i] += diagnoal[3][i]*6
    for i in range(5):
        a[2][i] += a[1][i]*1 
        diagnoal[2][i] += diagnoal[1][i]*1

    for i in range(5):
        a[0][i] += a[4][i]*(-2) 
        diagnoal[0][i] += diagnoal[4][i]*(-2)
    for i in range(5):
        a[0][i] += a[2][i]*(-2) 
        diagnoal[0][i] += diagnoal[2][i]*(-2)
    for i in range(5):
        a[1][i] += a[2][i]*(-1) 
        diagnoal[1][i] += diagnoal[2][i]*(-1)

    for i in range(5):
        a[1][i] += a[0][i]*1 
        diagnoal[1][i] += diagnoal[0][i]*1
    for i in range(5):
        a[1][i] += a[3][i]*1 
        diagnoal[1][i] += diagnoal[3][i]*1

    for i in range(5):
        a[3][i] += a[0][i]*1 
        diagnoal[3][i] += diagnoal[0][i]*1
    for i in range(5):
        a[3][i] += a[1][i]*(-1) 
        diagnoal[3][i] += diagnoal[1][i]*(-1)

    for i in range(5):
        a[2][i] += a[3][i]*(-1) 
        diagnoal[2][i] += diagnoal[3][i]*(-1)
    for i in range(5):
        a[1][i] += a[3][i]*1 
        diagnoal[1][i] += diagnoal[3][i]*1

    for i in range(5):
        a[1][i] += a[2][i]*1 
        diagnoal[1][i] += diagnoal[2][i]*1
    for i in range(5):
        a[0][i] += a[1][i]*(-1) 
        diagnoal[0][i] += diagnoal[1][i]*(-1)
    for i in range(5):
        a[0][i] += a[2][i]*2 
        diagnoal[0][i] += diagnoal[2][i]*2

    for i in range(5):
        a[1][i] += a[2][i]*1 
        diagnoal[1][i] += diagnoal[2][i]*1
    for i in range(5):
        a[0][i] += a[2][i]*(-1) 
        diagnoal[0][i] += diagnoal[2][i]*(-1)
    return a
#------------------------------------------------
# B is the decoder
B = np.linalg.inv(A)
C = my_inverse_matrix(A)
print(B)
print(C)
# z is the decoded message
z1 = np.dot(B, y1)
#print(z1)

z2 = np.dot(C, y1)
#print(z2)

x2 = np.array(
    [[6],
     [7],
     [8],
     [9],
     [10]])

y2 = np.dot(A, x2)
#print(y2)
z2 = np.dot(B, y2)
#print(z2)
w2 = np.dot(C, y2)
#print(w2)

# define a function to compute |a-b|
#------------------------------------------------
def abssum(a, b):
    """this function computes the norm of a-b"""
    result = 0
    for i in range(len(a)):
        result = result + abs(a[i]-b[i])
    return result[0]
#------------------------------------------------

abssum(x1, z1)
