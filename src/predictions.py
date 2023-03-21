
import itertools
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from SubpopulationsLib.Subpopulations import find_theta_sa, mixture_exponentials
from SubpopulationsLib.Metrics import MAPE
import warnings
from SubpopulationsLib.DataProcessing import create_SIR_data
from SubpopulationsLib.InfectiousModels import SIR

# All code is courtesy of Dr. Vega. It is just modified to add peak finding

warnings.filterwarnings("ignore")
def prediction(max_k, min_observ, start_date, end_date, country, file_path, lookup_table_path):
    results_dict = dict()
    results_dict[country] = dict()
    for i in range(max_k):
        results_dict[country]['Week ' + str(i+1)] = dict()
        results_dict[country]['Week ' + str(i+1)]['GT'] = list()
        results_dict[country]['Week ' + str(i+1)]['Gaussain_mix'] = list()
    
    S,I,R = create_SIR_data(country, file_path, lookup_table_path, start_date, end_date)
    indexes_weekly = np.arange(0,S.shape[0],7)
    S = S[indexes_weekly]
    I = I[indexes_weekly]
    R = R[indexes_weekly]
    data = I[1:]
    for i in range(min_observ, data.shape[0]-max_k):
        # Sets the parameters for the mixture of gaussians
        #print('Analyzing data up to the opbservation number: ' + str(i),end = '\r')
        sir = SIR()
        sir.train(S[:i], I[:i], R[:i])
        c_data = np.array(data[0:i])
        bias = np.min(c_data)
        norm_I = c_data - bias
        peaks, num_peaks = peak_finder(c_data)
        print('Number of peaks: ' + str(num_peaks))
        bounds_Gaussian = setup_mixture_gaussians(num_peaks,i)
        
        params_gaussian = find_theta_sa(bounds_Gaussian,norm_I, mixture_exponentials)
        # Make the predictions in advance
        S_last = S[i-1]
        I_last = I[i-1]
        R_last = R[i-1]
        for j in range(1,max_k+1):
            c_T = len(c_data)
            y_hat_gaussian = mixture_exponentials(params_gaussian, c_T+j) + bias
            S_last, I_last, R_last = sir.predict_next(S_last, I_last, R_last)
            results_dict[country]['Week ' + str(j)]['GT'].append(data[i-1 + j])
            results_dict[country]['Week ' + str(j)]['Gaussain_mix'].append(y_hat_gaussian)
    return results_dict


# Setup for the mixture of gaussians
def setup_mixture_gaussians(num_peaks,i):
    num_mixtures = num_peaks
    bounds_mu = (0,i+6)
    bounds_sigma = (1,6)
    bounds_coef = (0,300000)
    bound_list_Gaussian = [bounds_mu, bounds_sigma, bounds_coef]
    bounds_Gaussian = list()

    for element in bound_list_Gaussian:
        for i in range(num_mixtures):
            bounds_Gaussian.append(element)
    return bounds_Gaussian
    


# Needs fixing to work with smaller data sets
# finds the number of peaks and what the peaks are
def peak_finder(data):
    
    smoothed_data = sp.signal.savgol_filter(data, len(data)//2, 3)
    peaks = sp.signal.find_peaks(smoothed_data)[0]
    return peaks, len(peaks)

def visualize_results(results_dict, country, max_k):
    for k in range(1,max_k+1):
        print('Prediction' + str(k) + ' weeks in advance')
        gt = np.array(results_dict[country]['Week ' + str(k)]['GT'])
        predictions_gaussian = np.array(results_dict[country]['Week ' + str(k)]['Gaussain_mix'])
        print('MAPE Gaussian: ' + str(MAPE(gt, predictions_gaussian)))

        plt.figure()
        plt.plot(gt)
        plt.plot(predictions_gaussian)
        plt.legend(['Ground Truth', 'Gaussian Mix'])

# Driver code for this file. Sets the parameters and calls the functions
def driver():
    # Define the parameters
    start_date = '1/22/20'
    end_date = '1/22/21'
    country = 'Canada'
    file_path = "../Data/"
    lookup_table_path = "../Data/UID_ISO_FIPS_LookUp_Table.csv"
    dataframe = pd.read_csv(lookup_table_path)
    min_observ = 8
    max_k = 4
    country_names = np.unique(dataframe['Country_Region'].values)
    num_countries = len(country_names)
    results_dict = prediction(max_k, min_observ, start_date, end_date, country, file_path, lookup_table_path)
    visualize_results(results_dict, country, max_k)
    
driver()

