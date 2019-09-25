import numpy as np
from scipy.spatial.distance import pdist, squareform

def GaussianProcess(year,path):
    year_current = year
    npzfile = np.load(path)

    # read
    pred_out=npzfile['pred_out']
    real_out=npzfile['real_out']
    feature_out=npzfile['feature_out']
    year_out=npzfile['year_out']
    locations_out=npzfile['locations_out']
    index_out=npzfile['index_out']
    W = npzfile['weight_out']
    b = npzfile['b_out']
    W = np.concatenate((W,b))

    '''2 divide dataset'''

    # get train, validate, test index
    c1 = year_out==year_current
    # c2 = (index_out[:,0]==5)+(index_out[:,0]==17)+(index_out[:,0]==18)+(index_out[:,0]==19)+(index_out[:,0]==20)+(index_out[:,0]==27)+(index_out[:,0]==29)+(index_out[:,0]==31)+(index_out[:,0]==38)+(index_out[:,0]==39)+(index_out[:,0]==46)
    ind_test = np.where(c1)[0]
    print('shape of test set',ind_test.shape)

    c3 = year_out<year_current
    c4 = year_out>year_current-6
    ind_train = np.where(c3*c4)[0]
    print('shape of train set',ind_train.shape)

    '''4 normalize all features'''
    bias = np.ones([feature_out.shape[0],1])
    feature_out = np.concatenate((feature_out,bias),axis=1)

    locations_mean = np.mean(locations_out, axis=0,keepdims=True)
    locations_scale = np.amax(locations_out,axis=0)-np.amin(locations_out,axis=0)
    locations_out -= locations_mean
    locations_out /= locations_scale

    year_out = year_out[:,np.newaxis]
    year_mean = np.mean(year_out, axis=0,keepdims=True)
    year_scale = np.amax(year_out,axis=0)-np.amin(year_out,axis=0)
    year_out -= year_mean
    year_out /= year_scale

    real_out = real_out[:,np.newaxis]

    # split dataset
    feat_train = feature_out[ind_train,]
    feat_test = feature_out[ind_test,]
    Y_train = real_out[ind_train,]
    loc_train = locations_out[ind_train,]
    loc_test = locations_out[ind_test,]
    year_train = year_out[ind_train,]
    year_test = year_out[ind_test,]

    '''CNN baseline'''
    print ("The RMSE of CNN model is", np.sqrt(np.mean((real_out[ind_test,0]-pred_out[ind_test])**2)))
    '''CNN weight regression'''
    # print "The RMSE of regression, using CNN weight", np.sqrt(np.mean((real_out[ind_test,]-(np.dot(feat_test,W)))**2))
    print ("Mean Error of CNN is",np.mean(pred_out[ind_test]-real_out[ind_test,0]))

    '''
        Gaussian Prcoess Model 3,
        Linear GP as on page 28 of GP for machine learning
        kernel: spatial*time

    '''

    sigma=1
    l_s = 0.5
    l_t = 1.5
    noise = 0.1
    const = 0.01

    X_train = feat_train
    X_test = feat_test
    n1 = X_train.shape[0]
    n2 = X_test.shape[0]
    X = np.concatenate((X_train,X_test),axis=0)
    LOC = np.concatenate((loc_train,loc_test),axis=0)
    YEAR = np.concatenate((year_train,year_test),axis=0)
    pairwise_dists_loc = squareform(pdist(LOC, 'euclidean'))**2/l_s**2
    pairwise_dists_year = squareform(pdist(YEAR, 'euclidean'))**2/l_t**2

    n=np.zeros([n1+n2,n1+n2])
    n[0:n1,0:n1] += noise*np.identity(n1)
    kernel_mat_3 = sigma*(np.exp(-pairwise_dists_loc)*np.exp(-pairwise_dists_year))+n
    b = W
    B = np.identity(X_train.shape[1])

    print (l_s,l_t,noise,const)
    B /= const # B is diag, inverse is simplified
    K_inv = np.linalg.inv(kernel_mat_3[0:n1,0:n1])
    beta = np.linalg.inv(B+X_train.T.dot(K_inv).dot(X_train)).dot(
            X_train.T.dot(K_inv).dot(Y_train.reshape([n1,1]))+B.dot(b))
    Y_pred_3 = X_test.dot(beta) + kernel_mat_3[n1:(n1+n2),0:n1].dot(K_inv\
            ).dot(Y_train.reshape([n1,1])-X_train.dot(beta))

    RMSE_GP=np.sqrt(np.mean((Y_pred_3-real_out[ind_test,].reshape(Y_pred_3.shape))**2))
    ME_GP=np.mean(Y_pred_3[:,0]-real_out[ind_test,0])
    Average_GP=np.mean(Y_pred_3[:,0])
    print ("The RMSE of GP model is", RMSE_GP)
    print ("Mean Error of GP model is",ME_GP)
    # print "Average prediction of GP is",Average_GP

    '''If there is no bias'''
    print ("if there is no bias, the RMSE is")
    print ("CNN",np.sqrt(np.mean((real_out[ind_test,0]-pred_out[ind_test]+np.mean(pred_out[ind_test]-real_out[ind_test,0]))**2)))
    print ("GP",np.sqrt(np.mean((Y_pred_3-real_out[ind_test,].reshape(Y_pred_3.shape)-np.mean(Y_pred_3[:,0]-real_out[ind_test,0]))**2)))

    return (RMSE_GP,ME_GP, Average_GP)



