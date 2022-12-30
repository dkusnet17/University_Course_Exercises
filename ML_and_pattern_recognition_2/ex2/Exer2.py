# Week 1 - EX 2 -  Estimation theory
# LS and ML estimators
# Daniel Kusnetsoff


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def main():
    calc_LS_ML('Ex a',samples=100, noise=0, error=1.0)
    calc_LS_ML('Ex b',samples=400, noise=0, error=1.0) # tested 200,300 and 400
    calc_LS_ML('Ex c',samples=400, noise=2.0, error=1.0) #tested noise 0, 1.0, 2.0, 2.1, 2.2
    calc_LS_ML('Ex d',samples=400, noise=1.2, error=1.1) #tested 0, 1.0, 1.2, 1.3


def calc_LS_ML(title, samples, noise, error):

    LS_list, ML_list= f_zero(error=error, samples=samples, noise=noise)

    ls_f_hat = (np.argmin(LS_list)/samples)*(1/2) #
    ml_f_hat = (np.argmax(ML_list)/samples)*(1/2)

    n=np.arange(160) #(N) (from sinusoid())
    sin,sin_noise, A, phi=sinusoid(noise=noise)
    sin_hat = A*error*np.cos(2*np.pi*ml_f_hat*n + phi*error)

    print("done 1/4")
    draw(title,LS_list,ML_list, sin, sin_noise, sin_hat, noise, ls_f_hat, ml_f_hat) 
    #same amount param than draw() = 9



def sinusoid(noise):
    # Form a sinusoidal signal
    N = 160
    n = np.arange(N)
    f0 = 0.06752728319488948
    #x0 = A * np.cos(2 * np.pi * f0 * n+phi)
    # Add noise to the signal
    sigmaSq = 0.0 +noise# 1.2 #ex d?
    phi = 0.6090665392794814
    A = 0.6669548209299414
    #x = x0 + sigmaSq * np.random.randn(x0.size)
    # Estimation parameters moved to f_zero()
    #A_hat = A*1.0
    #phi_hat = phi*1.0
    #fRange = np.linspace(0, 0.5, 100)

    x0 = A * np.cos(2 * np.pi * f0 * n+phi) #moved here from original (did not do anything at original)
    x = x0 + sigmaSq * np.random.randn(x0.size) #moved here from original
    return x0, x, A, phi


def f_zero(noise, samples, error):
    #lists
    LS_list=list()
    ML_list=list()

    _, sin, A, phi=sinusoid(noise=noise)

    fRange=np.linspace(0,0.5,samples)
    
    A_hat=error*A
    phi_hat=error*phi


    for f_zero in fRange:
        ls_loss=0
        ml_tot=1
        for n, n1 in enumerate(sin):
            sin_hat=A_hat*np.cos(2*np.pi*f_zero*n+phi_hat)
            ls_loss +=np.square(n1-sin_hat)
            ml_tot *= stats.norm.pdf(sin_hat, n1)
    
        LS_list.append(ls_loss)
        ML_list.append(ml_tot)

    return LS_list, ML_list  




def draw(title, LS_list, ML_list,sin,sin_noise, sin_hat, f_hat, ls_f_hat, sigma2):
    #draw the plots
    fig, axs= plt.subplots(2,2)
    #fig.title(title)
    axs[0, 0].plot(LS_list)
    axs[0,0].set_title('LS')

    axs[0, 1].plot(ML_list)
    axs[0,1].set_title('ML')

    axs[1, 0].plot(sin)
    axs[1, 0].plot(sin_noise, 'r-')
    axs[1,0].set_title(f'Signal and noisy samples {f_hat}')

    axs[1, 1].plot(sin, 'r--')
    axs[1, 1].plot(sin_hat, 'b-')
    axs[1,1].set_title(f'True f0=0.0675 and estimated f0={sigma2}')

    plt.show()
    

if __name__ =="__main__":
    main()