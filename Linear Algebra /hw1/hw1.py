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
x2 = np.array(
    [[1],
     [0],
     [4],
     [0],
     [7]])

# y is the encoded message
y1 = np.dot(A, x1)
print(y1)

y2 = np.dot(A, x2)
print(y2)

def mydet(A):
    """"compute deteminant of A using cofactor expansion."""
    n = A.shape[0]
    if n == 1:
        return A[0][0]
    elif n == 2:
        return A[0][0]*A[1][1]-A[0][1]*A[1][0]
    else:
        det = 0
        for i in range(n):
            #compute minor M(0,i)
            if i==0:
                M = A[1:n, 1:n]
            elif i==n-1:
                M = A[1:n, 0:n-1]
            else:
                M = np.concatenate((A[1:n,0:i],A[1:n,i+1:n]),axis=1)

            # compute cofactor expansion
            # call mydet recursively to compute det(M)
            if i%2 == 0:
                det = det + A[0][i]*mydet(M)
            else:
                det = det - A[0][i]*mydet(M)
        return det

M41 = np.array(
    [[1, 0, 2, 2],
     [0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]])

M51 = np.array(
    [[1, 0, 2, 2],
     [0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]])

M12 = np.array(
    [[6, 1, 0, 0],
     [0, 0, 1, 0],
     [7, 0, 0, 1],
     [8, 0, 0, 1]])

M32 = np.array(
    [[0, 0, 2, 2],
     [6, 1, 0, 0],
     [7, 0, 0, 1],
     [8, 0, 0, 1]])

M42 = np.array(
    [[0, 0, 2, 2],
     [6, 1, 0, 0],
     [0, 0, 1, 0],
     [8, 0, 0, 1]])

M52 = np.array(
    [[0, 0, 2, 2],
     [6, 1, 0, 0],
     [0, 0, 1, 0],
     [7, 0, 0, 1]])

M23 = np.array(
    [[0, 1, 2, 2],
     [0, 0, 1, 0],
     [7, 0, 0, 1],
     [8, 0, 0, 1]])

M43 = np.array(
    [[0, 1, 2, 2],
     [6, 0, 0, 0],
     [0, 0, 1, 0],
     [8, 0, 0, 1]])

M53 = np.array(
    [[0, 1, 2, 2],
     [6, 0, 0, 0],
     [0, 0, 1, 0],
     [7, 0, 0, 1]])

M34 = np.array(
    [[0, 1, 0, 2],
     [6, 0, 1, 0],
     [7, 0, 0, 1],
     [8, 0, 0, 1]])

M45 = np.array(
    [[0, 1, 0, 2],
     [6, 0, 1, 0],
     [0, 0, 0, 1],
     [8, 0, 0, 1]])

M55 = np.array(
    [[0, 1, 0, 2],
     [6, 0, 1, 0],
     [0, 0, 0, 1],
     [7, 0, 0, 1]])


det_A = (A[3][0]*( ((-1)**(4+1)) * mydet(M41) )) + (A[4][0]*( ((-1)**(5+1)) * mydet(M51) ))

#now, need to find adjA

A41 = ((-1)**(4+1)) * mydet(M41) 
A51 = ((-1)**(5+1)) * mydet(M51)
A12 = ((-1)**(1+2)) * mydet(M12)
A32 = ((-1)**(3+2)) * mydet(M32)
A42 = ((-1)**(4+2)) * mydet(M42) 
A52 = ((-1)**(5+2)) * mydet(M52)
A23 = ((-1)**(2+3)) * mydet(M23)
A43 = ((-1)**(4+3)) * mydet(M43)
A53 = ((-1)**(5+3)) * mydet(M53) 
A34 = ((-1)**(3+4)) * mydet(M34)
A45 = ((-1)**(4+5)) * mydet(M45)
A55 = ((-1)**(5+5)) * mydet(M55)

adjA_improve = np.array(
    [[0, 0, 0, A41, A51],
     [A12, 0, A32, A42, A52],
     [0, A23, 0, A43, A53],
     [0, 0, A34, 0, 0],
     [0, 0, 0, A45, A55]])

A_inv = (1/det_A)*adjA_improve

B = np.linalg.inv(A)

print(B)
print(A_inv)

z1 = np.dot(B, y1)
print(z1)
z2 = np.dot(adjA_improve, y1)

z3 = np.dot(B, y2)
print(z3)
z4 = np.dot(adjA_improve, y2)
# define a function to compute |a-b|
#------------------------------------------------
def abssum(a, b):
    """this function computes the norm of a-b"""
    result = 0
    for i in range(len(a)):
        result = result + abs(a[i]-b[i])
    return result[0]
#------------------------------------------------

print(abssum(x1, z1))
print(abssum(x1, z2))

print(abssum(x2, z3))
print(abssum(x2, z4))
