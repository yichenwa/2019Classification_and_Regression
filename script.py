import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    d = len(X[0]);
    k = int(max(y));
    n = len(X);

    #m is a matrix like this:[[[] [] [] [] []],[[] [] [] [] []]]  in each [] is the same label X's same class value. like all labe1 X's first class value
    m = [[[]for i in range(k)] for j in range(d)];
    for i in range(n):
        for j in range(d):
            m[j][int(y[i])-1].append(X[i][j]);

    means = np.zeros(shape=(d,k));
    for i in range(d):
        for j in range(k):
            means[i][j]=np.mean(m[i][j]);
    #printmeans;
    
    m_value=[];
    X_copy=X.copy();
    for i in range(d):
        m_value.append(np.mean(means[i]));
    for i in range(n):
        for j in range(d):
            X_copy[i][j]-=m_value[j];
    #covmat=np.dot(X_copy.T, X_copy);
    covmat=np.cov(X.T);

    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    d = len(X[0]);
    k = int(max(y));
    n = len(X);

    m = [[[] for i in range(k)] for j in range(d)];
    for i in range(n):
        for j in range(d):
            m[j][int(y[i]) - 1].append(X[i][j]);

    means = np.zeros(shape=(d, k));
    for i in range(d):
        for j in range(k):
            means[i][j] = np.mean(m[i][j]);
    #print means;#same as LDA-YC

    covmats=[]
    for i in range(k):
        m_label = np.zeros(shape=(len(m[0][i]), d));
        for j in range(len(m[0][i])):
            for k in range(d):
                m_label[j][k] = m[k][i][j];
#         m_cov = np.dot(m_label.T, m_label);
        m_cov = np.cov(m_label.T);
        covmats.append(m_cov)

    return means,covmats

def ldaTest(means, covmat, Xtest, ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels
    k = len(means[1])
    n = len(Xtest)
    ypred = np.zeros(shape=(n,1))
    
    for i in range(n):
        mx = -1
        for j in range(k):
            res = Xtest[i] - means.T[j]
            resT = res.T
            invcov = np.linalg.inv(covmat)
            val1 = np.dot(resT, invcov)
            val = np.dot(val1, res)
            final = -1/2*val
            total = np.exp(final)
            if (mx < total):
                mx = total
                ypred[i] = j+1
    
    count = 0
    for i in range(n):
        if (ytest[i] == ypred[i]):
            count+=1
    
    acc = (count/n)*100
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels
    k = len(means[1])
    n = len(Xtest)
    d = len(Xtest[0])
    ypred = np.zeros(shape=(n,1))
    for i in range(n):
        mx = float('-inf')
        for j in range(k):
            res = Xtest[i] - means.T[j]
            resT = res.T
            invcov = np.linalg.inv(covmats[j])
            val1 = np.dot(res, invcov)
            val = np.dot(val1, resT)
            final = -1/2*val
            det = np.linalg.det(covmats[j])
            div = np.power(det, d/2)
            total = np.exp(final)/div
            if (mx < total):
                mx = total
                ypred[i] = j+1

    count = 0
    for i in range(n):
        if (ytest[i] == ypred[i]):
            count+=1
    
    acc = (count/n)*100
    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
    xnp = np.array(X)
    ynp = np.array(y)
    N = len(xnp)
    xT = np.dot(xnp.T, xnp)
    invxT = np.linalg.inv(xT)
    w = np.dot(np.dot(invxT, xnp.T), ynp)
    wt = np.array(w).T
    res = 0
    for i in range(N):
        res += np.square(ynp[i][0] - np.dot(wt, xnp[i]))
    res = res/N
    #print(res)
    # IMPLEMENT THIS METHOD                                                   
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1     
    xnp = np.array(X)
    ynp = np.array(y)
    xT = np.dot(xnp.T, xnp)
    d = len(X[0])
    val = lambd*np.identity(d)
    invxT = np.linalg.inv(xT+val)
    w = np.dot(np.dot(invxT, xnp.T), ynp)
    # IMPLEMENT THIS METHOD                                                   
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    xnp = np.array(Xtest)
    ynp = np.array(ytest)
    N = len(Xtest)
    res = 0
    wt = np.array(w).T
    for i in range(N):
        res += np.square(ynp[i][0] - np.dot(wt, xnp[i]))
    res = res/N
    # IMPLEMENT THIS METHOD
    mse = res
    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                              
    #equation  - -X.T*Y + w*X.T*X + lambd*w
    yres = np.dot(X, np.transpose(w))
    error_1 = np.subtract(y.reshape(y.size), yres)
    error_2 = np.multiply(error_1, error_1)
    error_3 = np.sum(error_2)
    error_4 = np.dot(w, np.transpose(w))
    error_5 = lambd*error_4
    final_error_6 = 0.5*np.add(error_3, error_5)
    error = final_error_6
    prt_1 = -1*np.dot(np.transpose(X), y)
    prt_2 = np.dot(w, np.dot(np.transpose(X), X))
    prt_3 = np.add(prt_1.reshape(prt_1.size), prt_2)
    prt_4 = lambd*w
    error_grad = np.add(prt_3, prt_4)
    # IMPLEMENT THIS METHOD
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xp - (N x (p+1)) 
    N = len(x)
    #making a new array of given shape.
    Xp = np.ones((N, p+1))
    #raising power of all all rows of a single column in Xp
    for i in range(1, p+1):
          Xp[:, i] = np.power(x, i) 
    # IMPLEMENT THIS METHOD
    return Xp

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))

#QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
plt.title('QDA')

plt.show()



# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

#Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
print("Question 3")
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.show()

# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
print("Question 4")
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()

# # Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(mses3)] # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()
