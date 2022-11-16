import numpy as np
import matplotlib.pyplot as plt
import random


##################################### Exercise 10.8 Calculate Value for Integral #####################################
'''Show that the probability distribution from which the sample points should be drawn is 1/(2*sqrt(x)) and derive the
transformation formula for generating random numbers between 0 and 1 from this distribution. Using the formula, sample
N = 1e6 random points, which evaluates the integral; answer is approximately 0.84.'''


############################ Part A ############################
# weight function
def w(x):
    return x**-0.5

# draw random values from 0 to 1 to get value of the weighted integral
def MeanValueMethod(a,b,N,f):
    funct = 0
    power = 0
    for i in range(1,N): 
        x = random.random()
        funct += f(x)
        power += abs(f(x))**2
        variance = (1/N)*power - ((1/N)*funct)**2
    return funct*(b-a)/N, variance

weightIntegral,variance = MeanValueMethod(0,1,1000,w)

# probability of distribution using weight function
def P(x):
    return w(x)/weightIntegral

# if you would like to see the visual representation of the distribution uncomment below:
n = 1000
X = np.linspace(1/n,1,n)
plt.plot(X,1/(2*np.sqrt(X)),'r')
plt.plot(X,P(X),'bo',markersize=1)
plt.legend(['w(x)/integral(w(x))','1/(2sqrt(x))'])
plt.title('distribution from which sample points are drawn')
plt.xlabel('x values')
plt.ylabel('probability')
plt.show()

############################ Part B ############################
N = 1e6

def f(x):
    return x**-0.5/(np.exp(x)+1)

def transformation(x):
    '''Obtained from using the transformation method: integrate p(x) from -inf to x(z),
    set this to equal to z, solve for x. z are the uniform random generated numbers.'''
    return x**2

def ImportanceSampling(N,f,w):
    distribution = 0
    for i in range(1,N):
        x = random.random()
        xi = transformation(x)
        distribution += f(xi)/w(xi)
    return distribution*weightIntegral/N                     # 2 is the value of the weighted integral

print('After sampling 1,000,000 random points, the computed value of the integral is',ImportanceSampling(int(N),f,w))