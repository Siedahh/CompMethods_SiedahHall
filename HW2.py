import gaussxw
import numpy as np
import matplotlib.pyplot as plt

############################################## PART A #################################################
#function
def f(t):
    return np.exp(-t**2)

#end points
a = 0
b = 3

#number of slices where steps are 0.1
N = int((b-a)/0.1)

def trapezoid(N,a,b):
    delx = (b-a)/N
    E = delx*(.5*f(a)+.5*f(b))
    for k in range(1,N):
        E += delx*(f(a+k*delx))
    #print("Integral using trapezoidal rule:", E)
    return E

# Gaussian quadrature
def gauss_quad(N,a,b):
    E = 0
    t,w = gaussxw.gaussxwab(N,a,b)  # get weights
    for k in range(N):
        E += f(t[k])*w[k]
    #print("Integral using Gaussian quadrature:", E)
    return t,E

#numerical solution
#gauss_quad(N,a,b)
E = trapezoid(N,a,b)
print("Integral using Gaussian quadrature:", E)
############################################## PART B #################################################

#new incremented b
x = 100
b = np.linspace(0,x,x)
E = []

#number of slices where steps are 0.1, due to new end point
for k in range(1,x+1):
    N_prime = int((k-a)/0.1)
    if N_prime is 0:
        E.append(0)
    else:
        #Ex = trapezoid(N_prime,a,k)
        #t,Ex = gauss_quad(N_prime,a,k)
        E.append(trapezoid(N_prime,a,k))
        #E.append(gauss_quad(N_prime,a,k))

plt.plot(b,E)
plt.show()