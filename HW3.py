import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.artist as artist

############################################## PART A #################################################
''' Two charges, +C and -C, 10cm apart, calculate the electric potential on a 1m x 1m square plane
surrounding the charges and passing through them. Calculate the electrostatic potential at 1cm spaced
points on the grid and make a visualization on the screen of the potential by using a density plot'''

# variables (position in units of cm)
q1,q2,x,y = 1,-1,10,0

# Electric potential
def E_potential(q1,q2,x,y):
    e0 = 1                                                      # permitivitty of free space, can use scipy.constants instead
    r1 = np.sqrt((x+5)**2+y**2)                                 # position of +C
    r2 = np.sqrt((x-5)**2+y**2)                                 # position of -C
    return (q1)/(4*np.pi*e0*(r1)) + (q2)/(4*np.pi*e0*(r2))      # sum of potentials

# x and y spatial ranges in units of cm
ax,bx,cx = -50,50,100
ay,by,cy = ax,bx,cx

# 2D array representing electric potential values in 2D space
E_pot_map = []

# filling 2D E_pot_map
for j in np.linspace(ay,by,cy):
    row=[]
    for i in np.linspace(ax,bx,cx):
        row.append(E_potential(1,-1,i,j))
    E_pot_map.append(row)
E_pot_map = np.array(E_pot_map)

# Visualization of potential in 1m x 1m grid
#fig,axis = plt.subplots(1,2,figsize = (12,6))
plt.imshow(E_pot_map, origin = 'lower', cmap = 'plasma',extent=[-50,50,-50,50])     # can also use 'RdBu'
plt.colorbar()      #(ax=axis[0])                                                   # norm=colors.LogNorm((vmin=E_pot_map.min(), vmax=E_pot_map.max()))
plt.clim(-0.005,0.005)
plt.xlabel(' x position (cm)')
plt.ylabel(' y position (cm)')
plt.show()

############################################## PART B #################################################
''' Calculate the partial derivatives of the potential with respect to x and y and hence find the
electric field in the xy plane. Make a visualization of the field. This is trickier than the potential,
because the electric field has both magnitude and direction. Matplotlib functions, quiver() and
streamplot() maybe useful.'''

# Partial derivatives w/ respect to x and y using central difference
def central_diff_2D(a,b,c):
    '''Partial differentiation in x and y
        a and b = limits, c = step size'''
    ddx = []
    ddy = []
    h = 1e-5
    for j in np.linspace(ay,by,cy):
        row_x = []
        row_y = []
        for i in np.linspace(ax,bx,cx):
            row_x.append((E_potential(1,-1,i+h/2,j)-E_potential(1,-1,i-h/2,j))/(-h))
            row_y.append((E_potential(1,-1,i,j+h/2)-E_potential(1,-1,i,j-h/2))/(-h))
        ddx.append(row_x)
        ddy.append(row_y)
    return np.array(ddx), np.array(ddy)


# Electric field (can redefine E to be every fifth position for visualization purposes)
Ex, Ey = central_diff_2D(ax,ay,cy)


# Visualization
#Q = [(-5, 0), (5, 0)]                                                                           # Charges (x-cord, y-cord)
#for q in Q:                                                                                     # charge locations
#    plt.text(q[0], q[1], 'o', color = 'y', fontsize=15, va='center', ha='center')
#    plt.text(q[0], q[1], 'o', color = 'y', fontsize=15, va='center', ha='center')

plt.text(-5, 0, '+', color = 'r', fontsize=15, va='center', ha='center', weight='bold')
plt.text(5,0, '-', color = 'r', fontsize=15, va='center', ha='center', weight='bold')

# Streamplot
color = np.log(np.sqrt(Ex**2 + Ey**2))                                                          # changes the color of the vector lines
X,Y = np.meshgrid(np.linspace(ax,bx,cy),np.linspace(ay,by,cy))
plt.streamplot(X,Y,Ex,Ey,color=color,density=1.2,cmap ='plasma')

# Quiver plot
#plt.colorbar(axis[1])
#plt.quiver(X,Y,Ex,Ey,scale=3)

# Draw circles for charges
#charge1 = plt.Circle((-5,0), 0.05, color = '#FF0000')
#charge2 = plt.Circle((5,0), 0.05, color = '#0000FF')
#artist(charge1)

plt.xlabel(' x position (cm)')
plt.show()