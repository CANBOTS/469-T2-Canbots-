import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import SubpopulationsLib.DataProcessing as dp
import sklearn.metrics as metrics
import scipy as sp
import datetime
from SubpopulationsLib.Subpopulations import mixture_exponentials, mixture_SIR, find_theta_sa
from SubpopulationsLib.Metrics import MAPE




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
    plt.ylabel("Number of Infected people")
    plt.title("Number of Infected people over time in " + country)
    return plt
#method which returns the peaks of the data
def get_peaks(data):
    smoothed_data =  sp.signal.savgol_filter(data, len(data)//2,5)
    peaks = sp.signal.find_peaks(smoothed_data)[0]

    if len(peaks) == 0:
        return 0,2

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

def fit_gaussian_model(data, params):
    
    bias = np.min(data)
    norm_I = data - bias
    T = len(norm_I)
    y_hat_gaussian = mixture_exponentials(params, T) + bias
    
    return y_hat_gaussian

def gaussian_params(data):
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
    return params_gaussian

def forecast_gaussian(data, params, steps):
    bias = np.min(data)
    norm_I = data - bias
    T = len(norm_I)
    y_hat_gaussian = mixture_exponentials(params, T) + bias

    for i in range(steps):
        c_T = len(y_hat_gaussian)
        new = mixture_exponentials(params, c_T + i) + bias

        y_hat_gaussian = np.append(y_hat_gaussian,[new[-1]])

    return y_hat_gaussian
    

def is_holdiday(date, country):
    if country == "Canada":
        holidays = {"01/01", "02/15", "04/02", "04/05", "05/24", "07/01", "08/02", "09/06", "10/11", "12/25"}
    elif country == "United States":
        holidays = {"01/01", "01/18", "02/15", "05/31", "07/05", "09/06", "10/11", "11/11", "11/25", "12/25"}
    date = date[0:-2]+"20"+date[-2:] #converts date to format that datetime can read
    date = datetime.datetime.strptime(date, "%m/%d/%Y")
    #check if there is a holday within two weeks after the date
    for i in range(1,15):
        if date.strftime("%m/%d") in holidays:
            return True
        date += datetime.timedelta(days=1)
    return False

#function to determine whether the end of the data set is a peak
def tail_peak(data):
    #assumes the data is already smoothed
    #check if the last 4 values are decreasing
    for i in range(4):
        if data[-1] < data[-i-1]:
            return 0
    return 1

#function to train a model to get the parameters for the next gaussian model based on the paramters of the previous gaussian curve
def get_next_params(params):
    mu = params[0] + 18 #assume that the next peak will be 18 weeks after the previous peak
    sigma = params[1] #assume that the next peak will have the same standard deviation as the previous peak
    coef = params[2]

    return mu, sigma, coef




    



def main():
    path = "./Data/"
    start_date = '7/30/20'
    end_date = '7/30/21'
    country = 'Canada'
    data = read_data(path, start_date, end_date, country)
    plot = plot_data(data, country)
    peaks, num_peaks = get_peaks(data)
    #sir = fit_sir_model(data)
    g_params = gaussian_params(data)
    
    gaussian = fit_gaussian_model(data, g_params)

    plot.plot(gaussian, color = 'red')
    print("Gaussian MAPE: ", MAPE(data, gaussian))
    #print("SIR MSE: ", mse(data, sir))    
 
    # # print("SIR MAPE: ", MAPE(data, sir))    
    #plot.plot(sir, color = 'green')
    plot.plot(peaks, data[peaks], 'bo')
    plot.show()


    
main()
