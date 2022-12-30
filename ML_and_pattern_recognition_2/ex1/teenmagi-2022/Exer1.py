
#Exercise 1 - Level test
#Team name in Kaggle: Daniel Kusnetsoff 
# Student number: last digits :192

import pickle
import csv
import numpy as np

from sklearn.neighbors import KNeighborsClassifier



def main():
    x_tr, y_tr, x_val = loadData()
    print("data load ok")

    model=KNeighborsClassifier(n_neighbors=6, algorithm='auto')

    model.fit(x_tr, y_tr)
    print("fitting model ok")

    y_pred = model.predict(x_val)
    print("predicting model ok")

    write_to_csv(y_pred)
    print("csv done")

def loadData():
    with open("C:/Users/danie/Data/training_x.dat", 'rb') as pickleFile:
        x_tr = pickle.load(pickleFile)
    with open("C:/Users/danie/Data/training_y.dat", 'rb') as pickleFile:
        y_tr = pickle.load(pickleFile)
    with open("C:/Users/danie/Data/validation_x.dat", 'rb') as pickleFile:
        x_val = pickle.load(pickleFile)

    
    amount_images_tr=len(y_tr)
    amount_images_val=len(x_val)

    
    x_tr[216805]=x_tr[216805][:,:, 0:3] #->wrong size/shape
    #(ValueError: could not broadcast input array from shape (8,8,3) into shape (8,8))

    x_tr_shape= np.asarray(x_tr).reshape(amount_images_tr, 64, 3)[:,:,0]
    y_tr_shape= np.asarray(y_tr)
    x_val_shape= np.asarray(x_val).reshape(amount_images_val, 64, 3)[:,:,0]

    return x_tr_shape, y_tr_shape, x_val_shape


def write_to_csv(y_pred):
    with open('submit111.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Id", "Class"])
        
        for i, pred in enumerate(y_pred): #
            writer.writerow([i+1, pred]) # bit slow but works
    print(f"csv done1") 


if __name__ == '__main__':
    main()

