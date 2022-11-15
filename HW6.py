import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


##################################### Exercise 10.3 Brownian Motion #####################################
'''Browninan motion of a particle in two dimensions. Peform random walk with one million steps and make
an animation of the position of the particle.'''


# Boundary
L = 101                                             # lattice size
dimension = 2
position = np.array([50,50])                        # LxL 
steps = int(1e5)
motion = np.empty([steps,dimension])

# random walk possible directions
walk = np.array([[1,0],[-1,0],[0,-1],[0,1]])        # [1,0]=right, [-1,0]=left, [0,-1]=down, [0,1]=up
rng = np.random.default_rng(seed=42)

# allowed movements at boundary
left_wall = np.array([[0,1],[0,-1],[1,0]])
right_wall = np.array([[0,1],[0,-1],[-1,0]])
top_wall = np.array([[0,-1],[-1,0],[1,0]])
bottom_wall = np.array([[0,1],[-1,0],[1,0]])

for step in range(steps):    
    if np.all(position==L):                         # top right corner
        position += rng.choice([[-1,0],[0,-1]])     
    elif np.all(position==0):                       # bottom left corner
        position += rng.choice([[1,0],[0,1]])          
    elif position[0]==L and position[1]==0:         # bottom right corner
        position += rng.choice([[-1,0],[0,1]])
    elif position[0]==0 and position[1]==L:         # top left corner
        position += rng.choice([[1,0],[0,-1]])
    elif position[0]==L:                            # right wall
        position += rng.choice(right_wall)
    elif position[1]==L:                            # top wall
        position += rng.choice(top_wall)
    elif position[0]==0:                            # left wall
        position += rng.choice(left_wall)
    elif position[1]==0:                            # bottom wall
        position += rng.choice(bottom_wall)
    else:                                           # inside grid
        position += rng.choice(walk)
    motion[step,:]=position


######################## Animation ########################
fig = plt.figure(figsize=(5,5))
ax = plt.axes(xlim=(0, 101), ylim=(0, 101))
particle = plt.Circle((50, 50), 1.5, fc='b')

def init():
    particle.center = (50, 50)
    ax.add_patch(particle)
    return particle,

def animate(i):
    x = motion[i,0]
    y = motion[i,1]
    particle.center = (x, y)
    return particle,

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=500, interval=20, blit=True)

#save as a gif
writergif = animation.PillowWriter(fps=30)
anim.save('HW6BrownianMotionMovie1.gif',writer=writergif)

# visualization of particle path after all steps
# inanimate variation
# plt.plot(motion[:,0],motion[:,1])

plt.title('Particle under the influence of Brownian motion')
plt.xlabel('x position (m)')
plt.ylabel('y position (m)')
plt.show()