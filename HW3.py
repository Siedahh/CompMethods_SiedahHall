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
    e0 = 1                                                      # permitivitty of free space set to 1
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
        row.append(E_potential(q1,q2,i,j))
    E_pot_map.append(row)
E_pot_map = np.array(E_pot_map)

############## Visualization of potential in 1m x 1m grid

# discrete color values for image
bounds=np.array([-1,-2e-2,-1e-3,-4e-4,-1e-4,1e-4,4e-4,1e-3,2e-2,1])
norm = colors.BoundaryNorm(bounds,275)

plt.imshow(E_pot_map, origin = 'lower',cmap = 'RdBu_r',extent=[ax/2,-ax/2,ay/2,-ay/2],norm = norm)
clb = plt.colorbar(format='%.0e')
plt.clim(bounds[0],bounds[bounds.size-1])
clb.set_label('Electric Potential')
plt.title('Electric Potential of a Positive and Negative Charge')
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
            row_x.append((E_potential(q1,q2,i+h/2,j)-E_potential(q1,q2,i-h/2,j))/(-h))
            row_y.append((E_potential(q1,q2,i,j+h/2)-E_potential(q1,q2,i,j-h/2))/(-h))
        ddx.append(row_x)
        ddy.append(row_y)
    return np.array(ddx), np.array(ddy)


# Electric field
Ex, Ey = central_diff_2D(ax,ay,cy)


############# Visualization of Electric Field

# positive and negative charge labels
plt.text(-5, 0, '+', color = 'r', fontsize=15, va='center', ha='center', weight='bold')
plt.text(5,0, '-', color = 'b', fontsize=15, va='center', ha='center', weight='bold')

color = np.log(np.sqrt(Ex**2 + Ey**2))                              # gradient color for E field
X,Y = np.meshgrid(np.linspace(ax,bx,cy),np.linspace(ay,by,cy))
plt.streamplot(X,Y,Ex,Ey,color=color,density=1.3,cmap ='Greys')     # Streamplot
clb=plt.colorbar()
clb.set_label('Electric Field')
plt.title('Electric Field of a Positive and Negative Charge')
plt.xlabel('x position (cm)')
plt.ylabel('y position (cm)')
plt.show()