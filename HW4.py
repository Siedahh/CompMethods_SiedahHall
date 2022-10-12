import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as const


############################################## The Lagrange Point #################################################
''' Write a program the uses Newton's method to solve for the distance r from the Earth to the L_1 point. Compute a
solution accurate to at least 4 significant figures'''

# constants
m = 7.348e22        # moon's mass
R = 3.844e8         # distance between Earth and moon
w = 2.662e-6        # angular velocity of the moon and the satellite
accuracy = 1e4      # significant figures for accuracy of guess

# functions for Newton's method 
def f(r):
    return (const.G.value * const.M_earth.value)/r**2 - (const.G.value * m)/(R-r)**2 - w**2*r

def f_prime(r):
    return (-2)*(const.G.value * const.M_earth.value)/r**3 - 2*(const.G.value * m)/(R-r)**3 - w**2

# plot polynomial to approximate roots
def plot_poly():
    r = np.arange(1e8,int(4e8),1e6)
    y = (const.G.value * const.M_earth.value)/r**2 - (const.G.value * m)/(R-r)**2 - w**2*r
    plt.plot(r,y)
    plt.show()

# Newton's Method
def Newtons_meth(r0):
    for k in range(int(1e5)):
        r_new = r0 - (f(r0) / f_prime(r0))
        diff = abs(r_new-r0)
        err = accuracy
        if err>diff:
            return r_new
            break
        r0 = r_new


if __name__ == "__main__":
    #plot_poly()        # use to approximate r0 for Newtons_meth(r0)
    print('The Lagrange point, L1, between the Earth and the moon is', Newtons_meth(1e8), 'm away from Earth')
