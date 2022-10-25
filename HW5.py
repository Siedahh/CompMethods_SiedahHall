import numpy as np 
import matplotlib.pyplot as plt
import dcst


#################################### Exercise 7.4 ######################################

# read data from dow.txt
closing_value = np.loadtxt('dow.txt')

# plot data on graph
plt.plot(closing_value)
plt.xlabel("Business Day")
plt.ylabel("Closing Value")
plt.title("Dow Jones Industrial Average (late 2006 - end 2010)")

def smooth_func(data,x):
    '''Takes percent value (eg. 90 for 90%, 13 for 13%) and returns less "noisy" data.
    The lower the percentage the lower the noise'''
    fft_coef = np.fft.rfft(data)                            # calculate coefficients of DFT
    remaining_fft_coeff = fft_coef.size * x/100             # choosing coefficients to keep
    fft_coef[int(remaining_fft_coeff):] = 0                 # set all coefficients but desired percentage to zero
    return np.fft.irfft(fft_coef)                           # inverse Fourier Transform

# smoothing the function
x = int(input("Please enter percentage of coefficients to keep: "))
new_closing_value = smooth_func(closing_value,x)

# plot on same graph
plt.plot(new_closing_value)
plt.legend(["raw data","smoothed data"])
plt.show()

print("\n\nCOMMENT: When setting a fraction of the coefficients to zero, the plot smoothes and is less noisy.\n\n")

#################################### Exercise 7.6 ######################################
####################################### PART A #########################################

# read data from dow2.txt
closing_value2 = np.loadtxt("dow2.txt")

# plot data on graph
plt.plot(closing_value2)
plt.xlabel("Business Day")
plt.ylabel("Closing Value")
plt.title("Dow Jones Industrial Average (2004 - 2008)")

# smoothing the function
x = 2
new_closing_value = smooth_func(closing_value2,x)

# plot on same graph
plt.plot(new_closing_value)
plt.legend(["raw data","smoothed data"])
plt.show()


####################################### PART B #########################################

def cos_smooth_func(data,x):
    '''Takes percent value (eg. 90 for 90%, 13 for 13%) and returns less "noisy" data.
    The lower the percentage the lower the noise'''
    dct_coef = dcst.dct(data)                               # calculate coefficients of DCT
    remaining_dct_coeff = dct_coef.size * x/100             # choosing coefficients to keep
    dct_coef[int(remaining_dct_coeff):] = 0                 # set all coefficients but desired percentage to zero
    return dcst.idct(dct_coef)                              # inverse Cosine Transform

# using DCT
x = int(input("Please enter percentage of coefficients to keep: "))
new_closing_value2 = cos_smooth_func(closing_value2,x)

# plot on same graph
plt.plot(closing_value2)
plt.xlabel("Business Day")
plt.ylabel("Closing Value")
plt.title("Dow Jones Industrial Average (2004 - 2008)")
plt.plot(new_closing_value2)
plt.legend(["raw data","smoothed data"])
plt.show()

#if __name__ == "__main__":
    