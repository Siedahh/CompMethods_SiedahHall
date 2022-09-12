import matplotlib.pyplot as plt
import numpy as np

#part a) deltoid curve
theta1 = np.linspace(0,2*np.pi,100)
x1 = 2*np.cos(theta1) + np.cos(2*theta1)
y1 = 2*np.sin(theta1) - np.sin(2*theta1)

#part b) polar plot galilean spiral
theta2 = np.linspace(0,10*np.pi,10000)
r = theta2**2
x2 = r*np.cos(theta2)
y2 = r*np.sin(theta2)

#part c) polar plot of Fey's Function
theta3 = np.linspace(0,24*np.pi,1000)
r = np.exp(np.cos(theta3))-2*np.cos(4*theta3)+(np.sin(theta3/12))**5
x3 = r*np.cos(theta3)
y3 = r*np.sin(theta3)

# plotting 3 plots in the same figure
figure, axis = plt.subplots(1, 3)

axis[0].plot(x1, y1)
axis[0].set_title("A) Deltoid Curve")
axis[0].set_xlabel('x')
axis[0].set_ylabel('y')

axis[1].plot(x2, y2)
axis[1].set_title("B) Galilean Spiral")
axis[1].set_xlabel('x')

axis[2].plot(x3, y3)
axis[2].set_title("C) Fey's Function")
axis[2].set_xlabel('x')

plt.show()