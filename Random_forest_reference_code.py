#!/usr/bin/env python
# coding: utf-8


import numpy as np
from numpy.linalg import norm
import glob
import random
from PIL import Image
import joblib
import time
from tqdm import tqdm

# Flags to specify if training/testing is to be executed
do_train = True
do_test = True

# Flag to specify if a model has to be saved and if testing should be done from saved model
# NOTE: Due to its large size trained model has not been saved in the submission
save_model = False


# Scaling factor for SRCNN
scaling_factor = 3

# size of square patch for whichSRF is to be trained
patch_size = 9


class RandomForest():
    def __init__(self, x, y, n_samples, n_trees, max_depth=8, min_leaf=5):
        self.n_samples, self.max_depth, self.min_leaf  = n_samples, max_depth, min_leaf
        self.trees = [self.create_tree(x,y) for i in range(n_trees)]
        
    def create_tree(self,x,y):
        idxs = np.random.permutation(len(x))[:self.n_samples]
        return DecisionTree(x[idxs], y[idxs], self.max_depth, self.min_leaf)
    
    def predict(self, x):
        return np.mean([t.predict(x) for t in self.trees], axis=0)
    
class DecisionTree():
    def __init__(self, x, y, depth, min_leaf=5):
        self.depth, self.min_leaf, = depth, min_leaf
        
        if(self.depth == 1 or len(x)<self.min_leaf):
            self.is_leaf = True
            self.split = None
            self.thresh = None
            self.lhs = None
            self.rhs = None
            self.weight = self.compute_weights(x,y)
        else:
            self.is_leaf = False        
            self.split, self.thresh = self.find_split(x,y)
            if self.split == None:
                self.is_leaf = True
                self.thresh = None
                self.lhs = None
                self.rhs = None
                self.weight = self.compute_weights(x,y)
                return
            id_lhs = np.where(x[:,self.split]<= self.thresh)
            id_rhs = np.where(x[:,self.split]> self.thresh)
            self.lhs = DecisionTree(x[id_lhs],y[id_lhs],self.depth-1,self.min_leaf)
            self.rhs = DecisionTree(x[id_rhs],y[id_rhs],self.depth-1,self.min_leaf)
        
    def compute_weights(self,XL,XH):
        [N,DL] = XL.shape
        L = 0.01
        temp = np.linalg.pinv(XL)
        temp = np.matmul(temp,XH)
        temp = np.matmul(XL.T,XL) + L*np.identity(DL)
        temp = np.linalg.inv(temp)
        temp = np.matmul(temp,XL.T)
        temp = np.matmul(temp,XH)
        W = temp.T
        return W
        
    def find_split(self,x,y):
        DL = x.shape[1]
        idx = random.sample(population=range(DL),k=DL//10)

        Q = np.inf
        split = None
        for id in idx:
            threshold = np.mean(x[:,id])
            sigma = x[:,id]<threshold
            Q_id = self.compute_q(x[sigma],y[sigma],x[np.invert(sigma)],y[np.invert(sigma)])
            if Q_id < Q:
                Q = Q_id
                split = id

        return split,threshold
    
    
    ### UNCOMMENT next bock to use ReF quality measure
#     #  Reconstruction-based objectiv (ReF)
#     def compute_e(self,XL_data,XH_data):
#         W = self.compute_weights(XL_data,XH_data)
#         XH_hat= np.matmul(W,XL_data.T).T
#         temp = np.square(XH_data - XH_hat)
#         temp = np.sum(temp)
#         XH_variance = temp/XH_data.shape[0]
#         XL_variance = np.sum(np.var(XL_data,axis=0))
#         E = XH_variance + XL_variance
#         return E

    
    ### UNCOMMENT next bock to use VaF quality measure
    # Reduction in Variance (VaF)
    def compute_e(self,XH_data,XL_data):
        XH_variance = np.sum(np.var(XH_data,axis=0))
        XL_variance = np.sum(np.var(XL_data,axis=0))
        E = XH_variance + XL_variance
        return E

    
    def compute_q(self,XL_Le,XH_Le,XL_Ri,XH_Ri):
        if(XL_Le.shape[0]==0 or XL_Ri.shape[0]==0):
            return np.inf
        E_Le = self.compute_e(XL_Le,XH_Le)
        E_Ri = self.compute_e(XL_Ri,XH_Ri)
        Q = XL_Le.shape[0]*E_Le + XL_Ri.shape[0]*E_Ri
        return Q

        
    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])
    
    def predict_row(self, xi):
        if self.is_leaf: 
            pred = np.matmul(self.weight,xi)
            return pred
        t = self.lhs if xi[self.split]<=self.thresh else self.rhs
        return t.predict_row(xi)

        
    

# Function to load training data
def load_train_data(train_path, scaling_factor, patch_size):
    train_hr = []
    train_lr = []
    
    train_set = glob.glob(train_path+"*.jpg")
    
    total_num = 45000
    image_num = total_num//len(train_set)
    
    for file in train_set:
        img = Image.open(file)
        img = img.convert('YCbCr')
        y, _, _ = img.split()
        H,W = y.size
        
        for idx in range(image_num):
            i = random.randrange(H-patch_size)
            j = random.randrange(W-patch_size)
            
            patch = y.crop((i,j,i+patch_size,j+patch_size))
            patch_lr = patch.resize((patch_size//scaling_factor,patch_size//scaling_factor),Image.BICUBIC)
            patch_lr = patch_lr.resize((patch_size,patch_size),Image.BICUBIC)
            
            
            patch = np.array(patch,dtype='int')
            if(np.var(patch)<100 and random.uniform(0,1)>0.5):                
                continue;
            patch_lr = np.array(patch_lr,dtype='int')
            patch = patch/255
            patch_lr = patch_lr/255
            patch = patch-patch_lr

            patch = patch.flatten()
            patch_lr = patch_lr.flatten()
            train_hr.append(patch)
            train_lr.append(patch_lr)
            
    train_hr = np.array(train_hr)
    train_lr = np.array(train_lr)        
    return train_lr, train_hr    



#Train the random forest
if(do_train):
    train_path = '../../data/SRF/train/'
    
    print("Loading training data...")
    train_lr, train_hr = load_train_data(train_path, scaling_factor, patch_size)

    print("Training in progress...")
    t0= time.time()
    srf = RandomForest(train_lr,train_hr,n_samples=15000, n_trees=12, max_depth=8)
    t1= time.time() - t0
    
    print("Training complete. Time duartion for training = {:.2f} seconds".format(t1))

    if(save_model):
        joblib_file = '../../results/SRF/models/SRF.pkl'
        joblib.dump(srf, joblib_file)


# Generate the test results with trained model
if(do_test):
    test_path = '../../data/SRF/test/'
    result_path = '../../results/SRF/images/'
    test_set = glob.glob(test_path+"*.jpg")
    
    if(save_model):
        joblib_file = '../../results/SRF/models/SRF.pkl'
        srf = joblib.load(joblib_file)

    psnr_list_bic = []
    psnr_list_srf = []
    
    with tqdm(total=len(test_set)) as t:
        for itr, file in enumerate(test_set):
            t.set_description("Generating test results")
            img = Image.open(file)

            img_bic = img.resize((img.size[0]//scaling_factor,img.size[1]//scaling_factor),Image.BICUBIC)
            img_bic = img_bic.resize((img.size),Image.BICUBIC)

            img_ycbcr = img_bic.convert('YCbCr')
            y, cb, cr = img_ycbcr.split()
            y_org = y

            y = np.array(y,dtype='int')
            y = y/255
            y = np.pad(y,patch_size-1,mode='symmetric')
            H,W = y.shape

            img_pred = np.zeros(y.shape)

            # Patch-wise construction of the high res image
            for i in range(0,H-patch_size+1):
                for j in range(0,W-patch_size+1):
                    lr = y[i:i+patch_size,j:j+patch_size]
                    lr = lr.reshape(1,-1)
                    pred = srf.predict(lr)
                    pred = pred+lr
                    pred = pred*255
                    pred = pred.clip(0,255)
                    pred = pred.reshape(patch_size,patch_size)

                    img_pred[i:i+patch_size,j:j+patch_size] += pred


            img_pred = img_pred/(patch_size*patch_size)
            img_pred = img_pred[patch_size-1:H-patch_size+1,patch_size-1:W-patch_size+1]

            # Combine Y with other channels and convert to RGB
            img_pred = np.array(img_pred,dtype='uint8')
            y2 = Image.fromarray(img_pred,mode='L')


            img_srf = Image.merge('YCbCr',(y2,cb,cr))
            img_srf = img_srf.convert('RGB')

            # Save results to disc
            img.save(result_path+str(itr)+'_orig.png')
            img_bic.save(result_path+str(itr)+'_bic.png')
            img_srf.save(result_path+str(itr)+'_srf.png')

            # Compute PSNR
            img = np.array(img,dtype='int')/255
            img_bic = np.array(img_bic,dtype='int')/255
            img_srf = np.array(img_srf,dtype='int')/255

            mse_bic = np.square(np.subtract(img,img_bic)).mean()
            mse_srf = np.square(np.subtract(img,img_srf)).mean()

            psnr_bic = 10 * np.log10(1/mse_bic)
            psnr_srf = 10 * np.log10(1/mse_srf)
            psnr_list_bic.append(psnr_bic)
            psnr_list_srf.append(psnr_srf)
            
            t.update(1)

    psnr_bic=np.array(psnr_list_bic).mean()
    psnr_srf=np.array(psnr_list_srf).mean()
    print("PSNR (Bicubic images) = {:.2f}".format(psnr_bic))
    print("PSNR (SRF images) = {:.2f}".format(psnr_srf))

