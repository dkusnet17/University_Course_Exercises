import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
from random import random


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


#X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
        #ValueError: operands could not be broadcast together with shapes (3072,) (32,32,3)
    #Y = np.array(Y)

#X = datadict["data"]
#Y = datadict["labels"]

#for i in range(X.shape[0]):
    # Show some images randomly
 #   if random() > 0.999:
 #       plt.figure(1);
 #       plt.clf()
 #       plt.imshow(X[i])
 #       plt.title(f"Image {i} label={label_names[Y[i]]} (num {Y[i]})")
 #       plt.pause(1)

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

def class_acc(pred,gt):
    temp_val=pred-gt
    temp_val_1=temp_val[temp_val==0]
    return (temp_val_1.size/pred.size)*100
    
    #print((temp_val_1.size/pred.size)*100)

def cifar10_classifier_random(x):
    return np.random.randint(0,9,x.shape[0])

def cifar10_classifier_1nn(x,trdata,trlabels):
    n=1
    pred=np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        distance=np.ones(trdata.shape[0])*np.inf
        smallest_distance=np.inf
        smallest_label=None
        for j in range (trdata.shape[0]):
            temp_distance=math.sqrt(((trdata[j]-x[i])**2).sum()) #euclidean
            distance[j]=temp_distance
            if temp_distance < smallest_distance:
                smallest_distance = temp_distance
                smallest_label = trlabels[j]
        pred[i]=smallest_label
        
    return pred

random_pred=cifar10_classifier_random(X)
rand_acc=class_acc(random_pred,Y)
print("random prediction accuracy = "+ str(rand_acc))

nn_pred=cifar10_classifier_1nn(X,train_X,train_Y)
nn_acc=class_acc(nn_pred,Y)
print("1NN prediction accuracy = " +str(nn_acc))

import socket
hostname = socket.gethostname()
ipaddr = socket.gethostbyname(hostname)
accuracy = str(nn_acc)
print(f"1-NN Accuracy: {accuracy} - Hostname: {hostname} - ipaddr: {ipaddr}")
