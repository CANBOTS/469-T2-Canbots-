import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import SubpopulationsLib.DataProcessing as dp
import sklearn.metrics as metrics
import scipy as sp
from SubpopulationsLib.Subpopulations import find_theta_sa
from SubpopulationsLib.Subpopulations import mixture_exponentials, mixture_SIR




#create a function to read the data from the csv file
def read_data(filename, start_date, end_date, country):
    S,I,R = dp.create_SIR_data(country, filename, './Data/UID_ISO_FIPS_LookUp_Table.csv', start_date, end_date)
    
    indexes_weekly = np.arange(0,S.shape[0],7)

    S = S[indexes_weekly]
    I = I[indexes_weekly]
    R = R[indexes_weekly]
    data = I[1:]
    return data


#create a function to plot the data using matplotlib
def plot_data(data, country):
    #create a figure

    plt.figure()
    #plot the data
    plt.plot(data)
    plt.xlabel("Time (weeks)")
    plt.ylabel("Number of infected people")
    plt.title("Number of infected people over time in " + country)
    return plt
#method which returns the peaks of the data
def get_peaks(data):
    smoothed_data =  sp.signal.savgol_filter(data, len(data)//2,3)
    peaks = sp.signal.find_peaks(smoothed_data)[0]
    return peaks, len(peaks)

def fit_sir_model(data):
    num_mixtures = get_peaks(data)[1]
    bound_S = (0,1E8)
    bound_beta = (0,1)
    bound_gamma = (0,1)
    bound_coef = (0,1000)
    bound_k = (0,50)
    bound_list_SIR = [bound_S, bound_beta, bound_gamma, bound_coef, bound_k]

    bounds_SIR = list()

    for element in bound_list_SIR:
        for i in range(num_mixtures):
            bounds_SIR.append(element)
    bias = np.min(data)
    norm_I = data - bias    
    params_SIR = find_theta_sa(bounds_SIR, norm_I, mixture_SIR)
    T = len(norm_I)
    y_hat_SIR = mixture_SIR(params_SIR, T) + bias
    
    return y_hat_SIR

def fit_gaussian_model(data):
    num_mixtures = get_peaks(data)[1]
    bounds_mu = (0,50)
    bounds_sigma = (1,6)
    bounds_coef = (0,300000)

    bound_list_Gaussian = [bounds_mu, bounds_sigma, bounds_coef]

    bounds_Gaussian = list()

    for element in bound_list_Gaussian:
        for i in range(num_mixtures):
            bounds_Gaussian.append(element)
    bias = np.min(data)
    norm_I = data - bias

    params_gaussian = find_theta_sa(bounds_Gaussian, norm_I, mixture_exponentials)
    T = len(norm_I)
    y_hat_gaussian = mixture_exponentials(params_gaussian, T) + bias
    return y_hat_gaussian



def main():
    path = "./Data/"
    start_date = '7/30/20'
    end_date = '7/30/21'
    country = 'Canada'
    data = read_data(path, start_date, end_date, country)
    plot = plot_data(data, country)
    peaks, num_peaks = get_peaks(data)
    sir = fit_sir_model(data)
    gaussian = fit_gaussian_model(data)
    plot.plot(gaussian, color = 'red')
    plot.plot(sir, color = 'green')
    plot.plot(peaks, data[peaks], 'bo')
    plot.show()


    
main()