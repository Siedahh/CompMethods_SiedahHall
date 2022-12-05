import numpy as np
import matplotlib.pyplot as plt


################################################### Exercise 8.7 ###################################################
'''A spherical cannon ball is shot from a cannon at ground level and the ball experiences a defined air resistance.
1. (do not need to show)
2. Change the two second order equations into four first order equations for a cannon ball and plot trajectory of
canonball.
3. Estimate total distance traveled and determine if canonball travels further if heavier or lighter.'''


################################################### PART 2 ###################################################
''' Using Runge-Kutta '''

# constants for equations of projectile motion under gravitational and drag forces
R = 0.8         # radius of canonball - m
row = 1.22      # density of air - kg/m^3
C = 0.47        # coefficient of drag
theta = 30      # angle from horizonatal - degrees
v = 100.0       # initial speed - m/s
g = 9.81        # acceleration due to gravity - m/s^2

r = np.array([0.0, 0.0, v*np.cos(theta*np.pi/180), v*np.sin(theta*np.pi/180)])     # x, y, v_x, v_y,

# vector function that operates on vector r
# conatines 4 first order ordinary differential equations
def f(r,t,m):
    xPosition = r[0]
    yPosition = r[1]
    xVelocity = r[2]
    yVelocity = r[3]
    f_xPosition = xVelocity
    f_yPosition = yVelocity
    f_xVelocity = -(np.pi*(R**2)*row*C*xVelocity*np.sqrt(xVelocity**2+yVelocity**2)/(2*m))
    f_yVelocity = -(g)-(np.pi*(R**2)*row*C*yVelocity*np.sqrt(xVelocity**2+yVelocity**2)/(2*m))
    return np.array([f_xPosition,f_yPosition,f_xVelocity,f_yVelocity])

# parameters for step size
a = 0
b = 1.4
N = int(1e4)

m = 1.0           # mass of canonball - kg

g_xPosition = []
g_yPosition = []
g_xVelocity = []
g_yVelocity = []

def RungeKutta(a,b,N,f,r,m):
    h = (b-a)/N

    tpoints = np.arange(a,b,h) 
    rpoints = []

    for t in tpoints: 
        rpoints.append(r)
        k1 = h*f(r,t,m)
        k2 = h*f(r+0.5*k1,t+0.5*h,m) 
        k3 = h*f(r+0.5*k2,t+0.5*h,m) 
        k4 = h*f(r+k3,t+h,m)
        r += (k1+2*k2+2*k3+k4)/6
        g_xPosition.append(r[0])
        g_yPosition.append(r[1])
        g_xVelocity.append(r[2])
        g_yVelocity.append(r[3])

RungeKutta(a,b,N,f,r,m)

# plot the trajectory
plt.plot(g_xPosition,g_yPosition)

ground = [0]*len(g_xPosition)

plt.plot(g_xPosition,ground,'--k')
plt.xlabel('horizontal position (m)')
plt.ylabel('height (m)')
plt.title('Projectile Motion of Canonball with Air Resistance')
plt.legend(['trajectory','ground'])
plt.show()


################################################### PART 3a ###################################################

# estimate total hoizontal distance traveled
for i in range(1,len(g_yPosition)):
    diff = abs(g_yPosition[i])
    err = 5e-4
    if err>diff:
        # print('Total vertical distance traveled: ', g_yPosition[i], 'meters')
        print('\nApproximate total horizontal distance traveled: ', g_xPosition[i], 'meters')
        break


################################################### PART 3b ###################################################

# does trajectory depend on mass?
m = [0.5,1.0,2.0,5.0,10.0]

# plot trajectory for random masses
for i in range(len(m)):
    g_xPosition.clear()
    g_yPosition.clear()
    g_xVelocity.clear()
    g_yVelocity.clear()

    b = [1.1,1.4,1.8,2.6,3.3]
    r = np.array([0.0, 0.0, v*np.cos(theta*np.pi/180), v*np.sin(theta*np.pi/180)])

    RungeKutta(a,b[i],N,f,r,m[i])
    plt.plot(g_xPosition,g_yPosition)


ground = [0]*len(g_xPosition)
plt.plot(g_xPosition,ground,'--k')
plt.xlabel('horizontal position (m)')
plt.ylabel('height (m)')
plt.title('Projectile Motion of Canonball with Air Resistance')
plt.legend(['0.5 kg','1.0 kg','2.0 kg','5.0 kg','10.0 kg','ground'])
plt.show()

# Yes, trajectory depends on mass
print('\nUnder the influence of air resistance, the greater the mass, the farther the canonball travels in the both the vertical and horizontal directions.\n')