import matplotlib.pyplot as plt
import numpy as np
    
#----------------------------------------------------    
def getData(fname):
    import csv

    # -------- inner function of getData ------------
    def getValue(s):
        if not s:     # an empty string
            return 0
        else:
            return float(s)

    # -------- open data file ---------------------     
    with open(fname, newline='', errors='ignore') as csvfile:
        # count the lines in CSV file
        N = sum(1 for line in csvfile)
        
        # prepare the space for data
        Y = np.zeros((N-1, 3))
        # array([[ 0.],[ 0.],[ 0.]
        #        [ 0.],[ 0.],[ 0.]
        #        [ 0.],[ 0.],[ 0.]]
        #               ...         )
        A = np.zeros((N-1, 6))

        # reset the file iterator
        csvfile.seek(0)

        # read the data from CSV file
        rows = csv.DictReader(csvfile)
        # the first line contains the titles
        i = 0
        for row in rows:
            Y[i,0] = getValue(row['O3_8hr'])
            Y[i,1] = getValue(row['PM10'])
            Y[i,2] = getValue(row['PM2.5'])
            
            # Get the data you want. Those are just examples.  

            A[i,0] = getValue(row['WindDirec'])
            A[i,1] = getValue(row['WindSpeed'])
            A[i,2] = getValue(row['Temperature'])
            A[i,3] = getValue(row['AtmosphericPressure'])
            A[i,4] = getValue(row['Moisture'])
            A[i,5] = getValue(row['Rainfall1day'])
            i = i + 1

    return i, A, Y

#----------------------------------------------------    
def view_O3_Relation(A, y):
    
    for i in range(6):
        plt.figure()
        plt.plot(A[:,i], y, 'ro') 
        plt.show()

def view_PM10_Relation(A, y):

    for i in range(6):
        plt.figure()
        plt.plot(A[:,i], y, 'ro')
        plt.show()
        
def view_PM2dot5_Relation(A, y):

    for i in range(6):
        plt.figure()
        plt.plot(A[:,i], y, 'ro')
        plt.show()

#----------------------------------------------------    
def build_O3_Model(n, A):

    B = np.zeros((n, 6))
    for i in range(n):

        # WindDirec
        x = A[i, 0]
        B[i, 0] = x*x*(43/100000) - x*(3/25) + 54

        # WindSpeed
        x = A[i, 1]
        B[i, 1] = x*x*x*(-17/100) + x*x*(207/100) - x*3 + 44

        # Temperature
        x = A[i, 2]
        B[i, 2] = x*x*x*(-1/50) + x*x*(141/100) - x*2 + 277

        # AtmosphericPressure
        if A[i, 3] > 1010 :
            B[i, 3] = x*x*(-88) + (178112)*x - 90124577;
        else :
            B[i, 3] = x*(5) + 35;

        # Moisture
        x = A[i, 4]
        B[i, 4] = x*x*x*(1/2500) - x*x*(9/100) + x*(37/5) - 100

        if(A[i,5]==0):
            B[i,5] = 0
        else:
            B[i,5] = A[i,5]
    return B

def build_PM10_Model(n, A):
    
    B = np.zeros((n, 6))
    
    for i in range(n):
        
        # WindDirec
        x = A[i, 0]
        B[i, 0] = x*x*(-11/12250) - (23/70)*x + 55
            
        # WindSpeed
        x = A[i, 1]
        B[i, 1] = x*(-10/19) + (575/19)
        
        # Temperature
        x = A[i, 2]
        B[i, 2] = x*x*(3) - (239/2)*x + (2285/2)
        
        # AtmosphericPressure
        if A[i, 3] > 1010 :
            B[i, 3] = x*x*(-88) + (178112)*x - 90124577;
        else :
            B[i, 3] = x*(5) + 35;
         
        # Moisture
        x = A[i, 4]
        B[i, 4] = x*(-2/9) + (415/9)
            
        # Rainfall1day
        if A[i, 5] <= 0.1 :
            x = A[i, 5]
            B[i, 5] = x*(-1900) + 200
        else :
            B[i, 5] = 0.5
    
    return B

def build_PM2dot5_Model(n, A):
    
    B = np.zeros((n, 6))
    
    for i in range(n):
        
        # WindDirec
        if A[i, 0] < 100 :
            x = A[i, 0]
            B[i, 0] = x*x*(1/240) - (5/8)*x + (215/6)
        elif ( A[i, 0] <= 250 and A[i, 0] >= 100 ) :
            x = A[i, 0]
            B[i, 0] = x*x*(1/1000) - (9/20)*x + 65
        else :
            x = A[i, 0]
            B[i, 0] = x*x*(11/5000) - (117/100)*x + (165)
            
        # WindSpeed
        x = A[i, 1]
        B[i, 1] = x*(-5/6) + (65/3)
        
        # Temperature
        if ( A[i, 2] >= 20 and A[i, 2] <= 30 ) :
            x = A[i, 2]
            B[i, 2] = x*x*(3/10) - (29/2)*x + (190)
        else :
            x = A[i, 2]
            B[i, 2] = x*(7/5) - (13)
        
        # AtmosphericPressure
        if A[i, 3] > 1010 :
            B[i, 3] = x*x*(-88) + (178112)*x - 90124577;
        else :
            B[i, 3] = x*(5) + 35;
         
        # Moisture
        x = A[i, 4]
        B[i, 4] = x*(-1/3) + (130/3)
            
        # Rainfall1day
        if A[i, 5] <= 0.5 :
            x = A[i, 5]
            B[i, 5] = x*(-54)+ 30
        elif ( A[i, 5] > 0.5 and A[i, 5] <= 2.5 ) :
            x = A[i, 5]
            B[i, 5] = x*(17/2) - (5/4)
        else :
            x = A[i, 5]
            B[i, 5] = x*(2/15) + (10/3)
    
    return B

#-------------------------------------------------------
def solve_O3_Model(B, y):
    # use this function to deal with normal equation 
    # build and solve the normal equation 
    B_transpose = np.transpose(B)
    BtB = B_transpose.dot(B)
    #print (BtB)
    BtBinv = np.linalg.inv(BtB)
    x = (BtBinv.dot(B_transpose)).dot(y)
    print (x)
    return x

def solve_PM10_Model(B, y):
    # use this function to deal with normal equation 
    # build and solve the normal equation 
    B_transpose = np.transpose(B)
    BtB = B_transpose.dot(B)
    #print (BtB)
    BtBinv = np.linalg.inv(BtB)
    x = (BtBinv.dot(B_transpose)).dot(y)
    print (x)
    return x

def solve_PM2dot5_Model(B, y):
    # use this function to deal with normal equation 
    # build and solve the normal equation 
    B_transpose = np.transpose(B)
    BtB = B_transpose.dot(B)
    #print (BtB)
    BtBinv = np.linalg.inv(BtB)
    x = (BtBinv.dot(B_transpose)).dot(y)
    print (x)
    return x

#-------------------------------------------------------
def validate_O3_Model(n, A, x, y):
    # using some training data to validate the model 
    import random
    
    # randomely select 100 records of data from A
    m = 100
    idx = random.sample(range(n), m)
    A1 = A[idx, :]
    y1 = y[idx]
    B1 = build_O3_Model(m, A1)
    
    z = B1.dot(x)
    # the figure shows the difference between prediction and the actual values 
    plt.figure()
    plt.plot(y1, z, 'bo')
    
    mse = np.linalg.norm(y1-z)/m
    
    return mse

def validate_PM10_Model(n, A, x, y):
    # using some training data to validate the model 
    import random
    
    # randomely select 100 records of data from A
    m = 100
    idx = random.sample(range(n), m)
    A1 = A[idx, :]
    y1 = y[idx]
    B1 = build_PM10_Model(m, A1)
    
    z = B1.dot(x)
    
    # the figure shows the difference between prediction and the actual values 
    plt.figure()
    plt.plot(y1, z, 'bo')
    
    mse = np.linalg.norm(y1-z)/m
    
    return mse

def validate_PM2dot5_Model(n, A, x, y):
    # using some training data to validate the model 
    import random
    
    # randomely select 100 records of data from A
    m = 100
    idx = random.sample(range(n), m)
    A1 = A[idx, :]
    y1 = y[idx]
    B1 = build_PM2dot5_Model(m, A1)
    
    z = B1.dot(x)
    
    # the figure shows the difference between prediction and the actual values 
    plt.figure()
    plt.plot(y1, z, 'bo')
    
    mse = np.linalg.norm(y1-z)/m
    
    return mse


# ------------- the prediction function ------------------
def predict_O3(n, C, x):
    # write your code here to predict O3 from unseen data C
    B = build_O3_Model(n, C)
    z = B.dot(x)
    return z

def predict_PM10(n, C, x):
    # write your code here to predict PM10 from unseen data C
    B = build_PM10_Model(n, C)
    z = B.dot(x)
    return z

def predict_PM2_5(n, C, x):
    # write your code here to predict PM2.5 from unseen data C
    B = build_PM2dot5_Model(n, C)
    z = B.dot(x)
    return z

# -------------the main body ------------------
N, A, Y = getData('airdata.csv')

# this is only for you to guess the relation between A and Y

#view_O3_Relation(A, Y[:,0])
#view_PM10_Relation(A, Y[:,1])
#view_PM2dot5_Relation(A, Y[:,2])

# build a linear regression model to predict O3
B = build_O3_Model(N, A)
#print (B)
E = build_PM10_Model(N, A)
#print (E)
D = build_PM2dot5_Model(N, A)
#print (D)

# all of the index of 0 in every row of Y
x = solve_O3_Model(B, Y[:,0])
y = solve_PM10_Model(E, Y[:,1])
z = solve_PM2dot5_Model(D, Y[:,2])

mse = validate_O3_Model(N, A, x, Y[:,0])
print ("validation errors of O3   :", mse)
print(mse)

mse2 = validate_PM10_Model(N, A, y, Y[:,1])
print ("validation errors of PM10 :", mse2)
print(mse2)

mse3 = validate_PM2dot5_Model(N, A, z, Y[:,2])
print ("validation errors of PM2.5:", mse3)
print(mse3)

# implement those three functions based on your computed models 
# TA will evaluate your models using the unseen data stored in C.
z1 = predict_O3(N, C, x)
z2 = predict_PM10(N, C, y)
z3 = predict_PM2_5(N, C, z)