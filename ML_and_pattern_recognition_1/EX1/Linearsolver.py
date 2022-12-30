# L inear s o l v e r
def my_linfit (x,y):
    a = (np.sum(x*y)/np.sum(x**2))-(np.sum(x)/np.sum(x**2))*  (((np.sum((x)*((np.sum(x*y))/(np.sum(x**2)))))-(np.sum(y)))/  (np.sum((x)*(np.sum(x)/(np.sum(x**2))-1))))
    b = (((np.sum((x)*((np.sum(x*y))/(np.sum(x**2)))))-(np.sum(y)))/  (np.sum((x)*(np.sum(x)/(np.sum(x**2))-1))))
    
    return a,b

# Main
import matplotlib.pyplot as plt
import numpy as np

x = np.random.uniform( -2,5,10 )
y = np.random.uniform( 0,3,10 )
a,b = my_linfit(x,y)
#a, b = np.polyfit(x,y,1)
plt.plot(x,y, 'kx' )
xp = np.arange( -2 , 5 , 0.1 )
plt.plot(xp, a * xp + b , 'r-' )
print (f"My fit:a={a} and b={b}" )
plt.show( )
