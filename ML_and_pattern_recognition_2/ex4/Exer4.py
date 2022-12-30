#Exercise 4
#Daniel Kusnetsoff

# Submit - Unfinished exercise :(



import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression



def main():
    #load .dat files
    x,y=data_load_and_norm()
    
    # Part 1. Logistic regression
    reg = LogisticRegression().fit(x,y)
    # coefficients for the prediction of each of the targets
    print("Logistic Regression Coefficients", reg.coef_)
    # do not use fit_intercept=none *
    reg1 = LogisticRegression(penalty='none', fit_intercept=False).fit(x, y)
    # coefficients for the prediction of each of the targets
    print("Logistic Regression Coefficients - No penalty", reg1.coef_)

    #Part 2. Logistic Regression with SSE

    sse_w= np.transpose([1,-1])
    learn_rate= 0.001
    sse_weights=[]
    sse_weights.append(sse_w)
    sse_accuracy=[]
    

    for i in range(100):
        for j in range(len(x)):
            sse_w = sse_w - learn_rate * reg_gradient_SSE(x[j], y[j], sse_w)
            sse_accuracy1, _ = prediction(sse_w, x, y)
        sse_accuracy.append(sse_accuracy1)
        sse_weights.append(sse_w)
    sse_accuracy = np.array(sse_accuracy)
    sse_weights = np.array(sse_weights)    


    #Part 3. Logistic Regression with ML

    ml_w=np.transpose([1,-1])
    ml_weights=[]
    ml_weights.append(ml_w)
    ml_accuracy=[]

    for i in range(100):
        
        ml_w = ml_w - learn_rate * x.Transpose.dot(y - sigmoid(x.dot(ml_w)))
        ml_accuracy1, _ = prediction(ml_w, x, y)
        ml_accuracy.append(ml_accuracy1)
        ml_weights.append(ml_w)
    ml_accuracy = np.array(ml_accuracy)
    ml_weights = np.array(ml_weights)   

    
    draw()

    print(f"{reg.coef_}")
    print(f"{reg.score(x,y)}")

    print(f"{reg1.coef_}")
    print(f"{reg1.score(x,y)}")

    print(f"SSE {sse_w}")
    print(f"Accuracy SSE {sse_accuracy}")

    print(f"ML {ml_w}")
    print(f"Accuracy ML {ml_accuracy}")


def data_load_and_norm():
    x=np.transpose(np.loadtxt('X.dat', unpack = True))
    y = np.loadtxt('y.dat', unpack=True)
   # normalization
    mean=np.mean(x[:,0])
    mean1=np.mean(x[:,1])

    for i in range(len(x)):
        x[i,0] = x[i,0] -mean
        x[i,1] = x[i,1] -mean1
    y[y==-1] = 0
    return x,y




def reg_gradient_SSE(y, x, w):

    return np.sum(-2*(y-sigmoid(np.sum(w*x)))*(1-sigmoid(np.sum(w*x)))*sigmoid(np.sum(w*x)))*x

def sigmoid(x):
    
    return (1/ (1+ np.exp(-x)))


def prediction(x, w, y, p_thresh=0.5):

    predict1 = (sigmoid(np.dot(x, w[:, np.newaxis])) >= p_thresh).astype(int).flatten() #-> error
    accuracy = np.mean(predict1 == y)
    return accuracy, predict1 



def draw(sse_weights, ml_weights, reg1, sse_accuracy, ml_accuracy,x,y):

    fig, axs = plt.subplots(2,1)

    axs[0].plot(sse_weights[:,0], sse_weights[:,1], 'o-')
    axs[0].plot(ml_weights[:,0], ml_weights[:,1], 'bo-')
    axs[0].plot(reg1.coef_[:,0], reg1.coef_[:,1], 'x')
    axs[0].set_title('Optimization')
    axs[0].set_xlabel('w1')
    axs[0].set_ylabel('w2')

    axs[1].plot(sse_accuracy, color= 'g')
    axs[1].plot(ml_accuracy, color= 'b')
    axs[1].axline(reg1.score(x,y), 0, 100, color='r')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Accuracy')

    plt.show()


    #accuracy_score(X, y)

    #return accuracy_score



if __name__ == '__main__':
    main()
