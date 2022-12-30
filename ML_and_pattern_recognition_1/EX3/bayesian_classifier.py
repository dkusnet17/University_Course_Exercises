#Bayesian classifier
import skimage.transform
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

#datadict = unpickle('/home/kamarain/Data/cifar-10-batches-py/data_batch_1')
datadict = unpickle('C:/Users/danie/Desktop/TUNI/ML-PatternRec/EX2/cifar-10-batches-py/test_batch')

X = datadict["data"]
Y = datadict["labels"]

#labeldict = unpickle('/home/kamarain/Data/cifar-10-batches-py/batches.meta')
labeldict = unpickle('C:/Users/danie/Desktop/TUNI/ML-PatternRec/EX2/cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]


X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
##np.resize
##df = pd.DataFrame({"A":[0, 1, 2, 3, 5, 9], "B":[11, 5, 8, 6, 7, 8], "C":[2, 5, 10, 11, 9, 8]}) ...
#Step 3 - Finding covariance. print(df.cov())


input_directory= ('C:/Users/danie/Desktop/TUNI/ML-PatternRec/EX2/cifar-10-batches-py/')
train_1= unpickle(input_directory+'data_batch_1')
train_2= unpickle(input_directory+'data_batch_2')
train_3= unpickle(input_directory+'data_batch_3')
train_4= unpickle(input_directory+'data_batch_4')
train_5= unpickle(input_directory+'data_batch_5')

#train_X=np.append(train_1["data"],train_2["data"],train_3["data"],train_4["data"],train_5["data"],axis=0)
train_X=np.append(train_1["data"],train_2["data"],axis=0)
train_X=np.append(train_X, train_3["data"],axis=0)
train_X=np.append(train_X, train_4["data"],axis=0)
train_X=np.append(train_X, train_5["data"],axis=0)


#print(type(train_1["labels"]))
train_Y=train_1["labels"]
train_Y.extend(train_2["labels"])
train_Y.extend(train_3["labels"])
train_Y.extend(train_4["labels"])
train_Y.extend(train_5["labels"])

train_Y = np.array(train_Y)
Y = np.array(Y)
print(len(train_Y))



def cifar10_color(X):
    resize(img, (1, 1)).astype(np.uint8)
    #mu[i] = C[i].mean(axis=0)
    #sigma[i] = np.sqrt(C[i].var(axis=0))
    #data = X[ Y == class]
    #np.var(data, axis=0)
    #np.cov(class_pictures, rowvar=False)

def cifar10_naivebayes_learn(Xp,Y): 
#that computes the normal distribution parameters(mu, sigma, p) for all ten classes


def cifar10_classifier_naivebayes(x,mu,sigma,p):
    #for row in x,
     #for cls in classes, 
        #for i in range(3)
        #X = X.reshape(len(X), 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
        #for i in range(X.shape[0]):
        #resized = resize(X[i],(1,1))


def cifar10_2x2_color(X)    



