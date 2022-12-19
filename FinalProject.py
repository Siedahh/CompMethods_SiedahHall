import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import time as t
import matplotlib.animation as animation

from mpl_toolkits.mplot3d import axes3d
from IPython.display import HTML


############################################################# PSEUDOCODE #############################################################
'''There are N identical particles in a perfectly reflective sphere. This sphere has a defined radius, as do the particles. The initial 
positions and directions of motion of the particles inside the sphere are uniformly randomized. A timestep is used to increment the 
positions of the particles until a specified (excited) particle has undergone a particle-particle collision. In this system all collisions 
are elastic. When the excited particle experiences a collision, the elapsed time from start is recorded and the events are looped for M 
number of excited particle collisions. The plot of the elapsed time for the events are given at the end of the procedure.'''

############################################################# NOTES #############################################################
'''The speed that the particles travel and the time steps associated in the calculations will give more precise measurements of a collision. 
It is possible that a particle can phase through another if the time step is too large. The time step should be determined by the radii 
of the particles and the average speed; it should be "very" small. The particle position error will determine the allowed distance error'''

'''runs very slow with only small time steps - used adaptive time steps: collision occurs at large time steps, then time is stepped 
backwards by one, and redone using a very small timestep - this speeds up the code significantly'''


'''If initial letter is capitalized, it is a function. If initial letter is lowercase, it is a variable.'''



class container:
    def __init__(self):
        self.radius = 10                                                            # radius of the sphere
        self.dimensions = 3                                                         # NEEED TO MAKE GLOBAL 
        self.position = np.zeros(self.dimensions, dtype=np.float64)                 # position of sphere is at the origin of the 3D Cartesian plane
        self.effectiveRadius = self.radius-p.radius                                 # particle has its own radius that will reflect off of container
        self.count = 0


    def Surface(self,particle,x,y,z):
        '''set spherical boundary conditions in cartesian coordinates for 
        particles inside the container | calls the function for bouncing
        off of spherical surface'''

        self.r = np.sqrt(x**2+y**2+z**2)                                            # location of the center of the particle with respect to the origin as r
        if self.r >= self.effectiveRadius:                                          # when particle is at or beyond the effective radius
            self.Bounce(particle,self.dimensions)                                   # bounce the particle off inside of the container


    def Bounce(self,particle,dimensions):
        '''changes direction of the particle to simulate bouncing from 
        container walls'''

        self.normalSpeed = 0       
        r = np.sqrt(p.position[particle,0]**2+p.position[particle,1]**2+p.position[particle,2]**2)
        self.tangentPlaneNormal = -1*(p.position[particle,:])/r                     # N = r/|r|, where r = (center of sphere) - (particle position), N is the unit vector normal from the surface of the sphere to the center
        
        for dim in range(dimensions):                                               # incrememnting through dimensions
            self.normalSpeed += self.tangentPlaneNormal[dim]*p.velocity[particle,dim]      # v = speed/component of particle velocity in direction of N, dot product                   
        
        initial_velocity = p.velocity[particle,:]
        final_velocity = 2*self.normalSpeed*self.tangentPlaneNormal 
        
        if (self.r >= self.effectiveRadius) and np.all(initial_velocity==abs(final_velocity)):     # if the particles have velocities that are opposite the bounce velocity - remove?
            print('did not occur')
            return

        p.velocity[particle,:] -= 2*self.normalSpeed*self.tangentPlaneNormal        # vf = vi + 2*v*N, final velocity pf particle



class particle:
    def __init__(self):

        ''' ******************* USER DEFINED VARIABLES - these variables are most likely to be changed *********************************** '''
        self.N = 5                                                                  # number of particles
        self.t_step_large = 1e-2                                                    # large time step (default until specified collision)
        self.t_step_small = 1e-7                                                    # small time step (after desired collision takes place and overlap is too large)
        self.loops = int(10)                                                      # number of desired experiments

        ''' ***************************************************************************************************************************** '''
        # "FIXED" VARIABLES
        self.radius = 0.7                                                           # radius of each particle
        self.D = 1                                                                  # diffusion constant
        self.dimensions = 3                                                         # number of dimensions
        self.speed = 1                                                              # user-defined speed
        self.t_elapse = 0                                                           # elapsed time
        self.collisionTime = []                                                     # time of collision for 'excited' particle - fills when code runs
        self.timeIndex = 0                                                          # used to initialize the collisionTime array        
        self.p0 = 0                                                                 # particle of interest / excited particle
        self.move = True                                                            # boolean used to tell the code to run
        self.bounceCount = 0
        self.position_error = self.radius/100                                       # allowed error for 'overlap' in position of particle


    def SphericalCartesian(self,i):
        '''random number generator for spherical coordinates, 
        converts to cartesian coordinates and stores in position array'''

        rng = np.random.default_rng()
        theta = rng.uniform(low = 0, high = np.pi)
        phi = rng.uniform(low = 0, high = 2*np.pi)
        r = rng.uniform(low = 0, high = int(sphere.effectiveRadius))
        self.position[i,0] = r*np.sin(theta)*np.cos(phi)
        self.position[i,1] = r*np.sin(theta)*np.sin(phi)
        self.position[i,2] = r*np.cos(theta)

    
    def DiffMag(self,i,j):
        '''calculates absolute difference between particles'''

        diff = self.position[i,:]-self.position[j,:] 
        return np.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2)


    def AdaptiveTime(self,r):
        '''adaptive time version 1 - checks for a particular distance
        between two particles in order to adjust time step'''

        if r <= 3*self.radius:
            self.t_step_wait = 1e-4
        else:
            self.t_step_wait = 1e-2


    def RandomPositions(self):
        '''creates random unique initial positions for particles uniform'''

        self.position = np.empty([self.N,self.dimensions])                          # position as a matrix of the number of particles by number of dimensions
        for i in range(self.N):                                                     # using 'for' loop to easily remove overlapping particles               
            self.SphericalCartesian(i)  

            # check for and reposition overlapping particles
            if i>0:
                for j in range(i):
                    mag_diff = self.DiffMag(i,j)
                    while mag_diff<2*self.radius:
                        self.SphericalCartesian(i)
                        mag_diff = self.DiffMag(i,j)


        
    def RandomVelocities(self):
        '''create random initial directions for particles uniform random 
        fill (x,y,x) velocities for each of the particles the range does 
        not matter, except (-) and (+) values are needed to have velocities 
        pointing in all directions the velocity unit vectors for the particles 
        are calculated a user defined speed is multipled so all particles have 
        the same speed and different orientations'''

        rng = np.random.default_rng()                                               # 43 seed for random numbers - produces same results for debugging | comment out when running for experiment
        self.velocity = np.empty([self.N,self.dimensions])                          # velocity as a matrix of the number of particles by number of dimensions
        for i in range(self.N):
            magnitude = 0
            magnitude_sqr = 0
            for j in range(self.dimensions):
                self.velocity[i,j]=rng.uniform(low = -1*int(sphere.effectiveRadius), 
                high = int(sphere.effectiveRadius))
                magnitude_sqr += self.velocity[i,j]**2                              # |v|^2
            magnitude = np.sqrt(magnitude_sqr)                                      # |v|
            self.velocity[i,:] = self.speed*self.velocity[i,:]/magnitude            # velocity = speed * vector(v)/|v|


    def Initiate(self):
        '''initiate random velocities and positions'''

        self.RandomPositions()
        self.RandomVelocities()


    def Run(self):
        '''particles move in time'''
        
        for running in range(int(self.loops)):
            self.t_elapse = 0                                                       # need to set t_elapse to zero so it does not get larger with each run
            self.t_step = self.t_step_large
            self.move = True
            self.count = 0
            self.observed_position = []
            self.Initiate()
            self.animate_pos = []

            while self.move == True:
                self.t_elapse += self.t_step                                        # increment time
                self.position += self.velocity*self.t_step                          # new position after some time and velocity   
                self.count += 1               
                self.animate_pos.append(self.position.tolist())                     # append nparray to list (not sure how many collision, so cannot size array properly)

                # particle - particle interactions
                if self.N>1:
                    for i in range(self.N):                                         
                        for j in range(i+1,self.N):
                            self.r = self.DiffMag(i,j)

                            if self.r<=2*self.radius:                               # when distance between particles is twice or less than the radius
                                if (i == self.p0) or (j == self.p0):                # if i is the chosen particle experiences collision
                                    self.hitter = j

                                    if self.r<(2*self.radius-self.position_error):
                                    # finding time with less roundoff error by using smaller timestep
                                        print('running with a smaller timestep')
                                        self.t_elapse -= self.t_step
                                        self.position[i,:] -= self.velocity[i,:]*self.t_step
                                        self.position[j,:] -= self.velocity[j,:]*self.t_step
                                        self.t_step = self.t_step_small

                                        while self.move == True :
                                            self.t_elapse += self.t_step
                                            self.position[i,:] += self.velocity[i,:]*self.t_step
                                            self.position[j,:] += self.velocity[j,:]*self.t_step
                                            self.r = self.DiffMag(i,j)
                                            self.animate_pos.append(self.position.tolist())
                                            if self.r<=2*self.radius:
                                                break

                                    self.collisionTime.append(self.t_elapse)
                                    self.move = False

                                    # record the collision time to an array
                                    self.timeIndex += 1
                                    print("collision", self.timeIndex)
                                    self.move = False
                                    break
                                self.Bounce(i,j)                                    # call bounce function if particle of interest does not collide
                        if self.move == False:
                            break
                    if self.move == False:
                        break

                # particle - container interactions
                if self.move == True:
                    for i in range(self.N):
                        sphere.Surface(i,self.position[i,0],self.position[i,1],self.position[i,2])


    def Bounce(self,a,b):
        '''negligible particles bounce off of one another | changes due to 
        conservation of momentum and energy'''

        # midpoint between particles and difference vectors between centers and midpoint
        self.midpoint = (self.position[a,:]+self.position[b,:])/2                   
        diff_a = self.position[a,:] - self.midpoint
        diff_b = self.position[b,:] - self.midpoint
        diff_a_sqrd = 0
        diff_b_sqrd = 0

        # magnitude of difference vectors, |r|
        for dim in range(self.dimensions):
            diff_a_sqrd += diff_a[dim]**2
            diff_b_sqrd += diff_b[dim]**2
        r_a = np.sqrt(diff_a_sqrd)                                                  # |r_a|
        r_b = np.sqrt(diff_b_sqrd)                                                  # |r_b|

        # use midpoint between the particles to find normal unit vector to tangent plane
        self.tangentPlaneNormal_a = diff_a/r_a                                      # N = r_a/|r_a|
        self.tangentPlaneNormal_b = diff_b/r_b                                      # N = r_b/|r_b|

        # speeds for both particles in normal directions
        self.normalSpeed_a = 0
        self.normalSpeed_b = 0
        for dim in range(self.dimensions):
            self.normalSpeed_a += self.tangentPlaneNormal_a*p.velocity[a,dim]       # dot product
            self.normalSpeed_b += self.tangentPlaneNormal_b*p.velocity[b,dim]

        # components of velocities in direction of normal
        a_velocity = self.normalSpeed_a*self.tangentPlaneNormal_a
        b_velocity = self.normalSpeed_b*self.tangentPlaneNormal_b

        # swap velocities for identical particles in normal direction, then change final velocity
        velocity_a_normal = self.normalSpeed_a*self.tangentPlaneNormal_a
        velocity_b_normal = self.normalSpeed_b*self.tangentPlaneNormal_b
        p.velocity[a,:] -= (velocity_a_normal - velocity_b_normal)
        p.velocity[b,:] -= (velocity_b_normal - velocity_a_normal)

        speed_a = np.linalg.norm(p.velocity[a,:])
        speed_b = np.linalg.norm(p.velocity[b,:])


    def Collision(self):
        '''choose a particle to watch and determine its first particle-particle interaction'''
        self.p0Position = self.position[self.p0,:]
        self.p0Velocity = self.velocity[self.p0,:]


    def Animate(self):
        '''animation of last experiment (loop) from start to collision'''
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(projection='3d')
        plt.axis('off')
        ax.set_box_aspect([1,1,1])                                                                      # ratio all three axes to 1
        posit = np.asarray(self.animate_pos)

        def init():
            mol = ax.scatter(posit[0,0,0],posit[0,0,1],posit[0,0,2],s = 100, marker = 'o',color='k')
            for j in range(1,p.N):
                mol = ax.scatter(posit[0,j,0],posit[0,j,1],posit[0,j,2],s=200, marker = 'o',color='b',alpha=0.1)
            return mol,

        def update(i):
            ax.cla()                                                                                    # clear plot
            ax.grid(False)                                                                              # remove grid
            ax.set_facecolor((0.1,0.1,0.1))
            plt.axis('off')                                                                             # remove axes

            mol = ax.scatter(posit[i,self.p0,0],posit[i,self.p0,1],posit[i,self.p0,2],                  # particle of interest (POI)
            s=100, marker='o',color='pink',alpha=1,edgecolors='none')                                   # s = np.pi*p.radius**2
            mol = ax.scatter(posit[i,self.hitter,0],posit[i,self.hitter,1],posit[i,self.hitter,2],      # particle that collides with POI
            marker = 'o',color='pink')

            for j in range(1,p.N):
                mol = ax.scatter(posit[i,j,0],posit[i,j,1],posit[i,j,2], s=100,marker = 'o',            # all non-POI
                color='purple',alpha=0.5,edgecolors='none')

            u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
            ax.plot_wireframe(10*np.sin(v)*np.cos(u),10*np.sin(v)*np.sin(u),10*np.cos(v),
            alpha =.1,color=(.5,.5,.5))                                                                 # spherical container
            #ax.plot_surface(10*np.sin(v)*np.sin(u),10*np.sin(v)*np.cos(u),10*np.cos(v),alpha =.1)      # for surface plot instead of wireframe

            ax.plot(posit[i:,self.p0,0],posit[i:,self.p0,1],posit[i:,self.p0,2],
            color='pink',linewidth=.75,linestyle='--')
            ax.plot(posit[i:,self.hitter,0],posit[i:,self.hitter,1],posit[i:,self.hitter,2],
            color='purple',linewidth=.75,linestyle='--')

            return mol,

        steps = np.shape(posit)[0]
        if steps>1:
            print('\nframes in animation:',steps)
            incr = int(steps/100)
            anim = animation.FuncAnimation(fig,update,frames=np.arange(0,steps,incr),interval=20)
            #print(anim.to_html5_video())
            plt.show()

        def save_gif():
            '''save as a gif of animation - currently runs very slow, prefer to screen shot the video'''
            writergif = animation.PillowWriter(fps=30)
            anim.save('ParticleCollision.gif',writer=writergif)

        # save_gif()


    def Stats(self):
        '''printing experiment variables and statistics | called after Run()'''

        print('\nNumber of particles:',self.N)
        print('Number of experiments:',self.loops)
        print('Time Step:', self.t_step)
        print('AVERAGE TIME OF COLLISION:',np.mean(self.collisionTime))
        print("MEDIAN TIME OF COLLISION",np.median(self.collisionTime))

        # scatter plot
        #axis[0].scatter(np.linspace(0,p.loops,p.loops),p.collisionTime)
        #axis[0].set_xlabel('experiment no.')
        #axis[0].set_ylabel('elapsed time')

        # histogram
        nBins = 100
        plt.hist(p.collisionTime,bins=nBins)             
        plt.xlabel('elapsed time')
        plt.ylabel('counts')   
        #mode_index = n.argmax()                                                                     # print mode (additional statistics depends on axis[1] plot)
        #print('Mode Bin range:(' + str(bins[mode_index]) + ',' + str(bins[mode_index+1]) + ')')        
        #print('Mode Time of Collisions:'+ str((bins[mode_index] + bins[mode_index+1])/2))

        plt.show()

    def SaveFile(self):
        '''output raw collision data into a file'''
        
        np.savetxt('collisionTimeData_'+str(p.N)+'particles_'+str(p.loops)+'experiments.txt', self.collisionTime)
        print('File saved')



if __name__=="__main__":

    p = particle()
    sphere = container() 
    print('N =',p.N)
    print('running',p.loops,'experiments...')                                                            
    p.Run()                                                                             # run the experiments
    p.Stats()                                                                           # display the stats
    #p.SaveFile()                                                                        # save the collision data for later analysis
    p.Animate()                                                                         # animate the last collision




