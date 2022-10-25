import numpy as np 
import matplotlib.pyplot as plt
import dcst
import argparse


#################################### Exercise 7.4 ######################################
#################################### Exercise 7.6 ######################################
##################################### PART A / B #######################################


def fft_smooth_func(data,x):
    '''Takes percent value (eg. 90 for 90%, 13 for 13%), x, and returns less "noisy" data.
    The lower the percentage the lower the noise'''
    fft_coef = np.fft.rfft(data)                                                # calculate coefficients of DFT
    remaining_fft_coeff = fft_coef.size * x/100                                 # choosing coefficients to keep
    fft_coef[int(remaining_fft_coeff):] = 0                                     # set all coefficients out of desired percentage to zero
    return np.fft.irfft(fft_coef)                                               # inverse Fourier Transform


def dct_smooth_func(data,x):
    '''Takes percent value (eg. 90 for 90%, 13 for 13%), x, and returns less "noisy" data.
    The lower the percentage the lower the noise'''
    dct_coef = dcst.dct(data)                                                   # calculate coefficients of DCT
    remaining_dct_coeff = dct_coef.size * x/100                                 # choosing coefficients to keep
    dct_coef[int(remaining_dct_coeff):] = 0                                     # set all coefficients out of desired percentage to zero
    return dcst.idct(dct_coef)                                                  # inverse Cosine Transform


def plot_transform_comparison(dowfilename,transform,percent_to_keep):
    '''Takes specified dow file, transformation type, and percentage of transformation coefficients to keep, 
    then plots the comparison of the original and smoothed data.'''

    # loading txt file                        
    if dowfilename == 'dow2':
        closing_value = np.loadtxt('dow2.txt')
        plt.title("Dow Jones Industrial Average (2004 - 2008)")
    if dowfilename == 'dow':
        closing_value = np.loadtxt('dow.txt')
        plt.title("Dow Jones Industrial Average (late 2006 - end 2010)")

    # plotting raw closing values
    plt.plot(closing_value)                                                     # original closing values
    plt.xlabel("Business Day")
    plt.ylabel("Closing Value")

    # smoothin closing_values based on chosen transformation type
    if transform == 'dft':
        new_closing_value = fft_smooth_func(closing_value,percent_to_keep)      # using fft
    if transform == 'dct':
        new_closing_value = dct_smooth_func(closing_value,percent_to_keep)      # using dct
  
    # plotting new closing values
    plt.plot(new_closing_value)                                                 # smoothed closing values
    plt.legend(["raw data","smoothed data"])
    plt.show()


# Command line options
parser = argparse.ArgumentParser("Smooth the Dow closing values.")
parser.add_argument("dowfile", choices=['dow','dow2'], help="run with dow or dow2")
parser.add_argument("transform", choices=['dft','dct'], help="run with dft or dct")                                                                           
parser.add_argument("--percent","--p", type=int, default=10, help="percentage (integer value) of Fourier coefficients to keep, default is 10")
args = parser.parse_args()

dow_file = args.dowfile
transform_type = args.transform
percent_val = args.percent


if __name__ == "__main__":
    plot_transform_comparison(dow_file,transform_type,percent_val)
    print("\nCOMMENT: When setting a fraction of the coefficients to zero, the plot smoothes and is less noisy.\n")

