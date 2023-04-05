

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from SubpopulationsLib.Subpopulations import find_theta_sa, mixture_exponentials, mixture_SIR
from SubpopulationsLib.Metrics import MAPE
import warnings
from SubpopulationsLib.DataProcessing import create_SIR_data
from SubpopulationsLib.InfectiousModels import SIR

# All code is courtesy of Dr. Vega. It is just modified to add peak finding
warnings.filterwarnings("ignore")
def peak_finder(data):
    smoothed_data = sp.signal.savgol_filter(data, 5, 3)
    peaks = sp.signal.find_peaks(smoothed_data)[0]
    num_peaks = len(peaks)
    if data[-1] >= data[-2]:
        num_peaks = num_peaks + 1
    return num_peaks


def main():
    # Determine the dates to use
    start_date = '7/30/20'
    end_date = '7/30/21'

    # Set the path to the datasets
    file_path = './Data/'
    lookup_table_path = './Data/UID_ISO_FIPS_LookUp_Table.csv'

    # Set the minimum number of observations that should be available for the experiment
    min_observ = 5

    # Determine the maximum horizon for forecasting (in weeks)
    max_k = 4

    # Get the name of all available countries in the dataset
    dataframe = pd.read_csv(lookup_table_path)

    country_names = np.unique(dataframe['Country_Region'].values)
    num_countries = len(country_names)

    country_name = 'Canada'
    results_dict = dict()
    results_dict[country_name] = dict()

    for i in range(max_k):
        results_dict[country_name]['Week ' + str(i+1)] = dict()
        results_dict[country_name]['Week ' + str(i+1)]['GT'] = list()
        results_dict[country_name]['Week ' + str(i+1)]['Gaussian_mix'] = list()
        results_dict[country_name]['Week ' + str(i+1)]['SIR_mix'] = list()

    S,I,R = create_SIR_data(country_name, file_path, lookup_table_path, start_date, end_date)
    indexes_weekly = np.arange(0,S.shape[0],7)
    S = S[indexes_weekly]
    I = I[indexes_weekly]
    R = R[indexes_weekly]
    data = I[1:]
    for i in range(min_observ,data.shape[0]-max_k):
        sir = SIR()
        sir.train(S[0:i], I[0:i], R[0:i])
        c_data = np.array(data[0:i])
        
        bias = np.min(c_data)
        norm_I = c_data - bias

        num_peaks = peak_finder(c_data)

        bounds_mu = (0,i+6)
        bounds_sigma = (1,6)
        bounds_coef = (0,300000)

        bound_list_Gaussian = [bounds_mu, bounds_sigma, bounds_coef]

        bounds_Gaussian = list()

        for element in bound_list_Gaussian:
            for j in range(num_peaks):
                bounds_Gaussian.append(element)

        # For the SIR model
        bound_S = (0,1E8)
        bound_beta = (0,1)
        bound_gamma = (0,1)
        bound_coef = (0,1000)
        bound_k = (0,i+6)
        bound_list_SIR = [bound_S, bound_beta, bound_gamma, bound_coef, bound_k]

        bounds_SIR = list()

        for element in bound_list_SIR:
            for j in range(num_peaks):
                bounds_SIR.append(element)

        # Fit with mixtuers of models
        params_gaussian = find_theta_sa(bounds_Gaussian, norm_I, mixture_exponentials)
        params_SIR = find_theta_sa(bounds_SIR, norm_I, mixture_SIR)

        # -----------------------------------------
        # Make the predictions in advance
        S_last = S[i-1]
        I_last = I[i-1]
        R_last = R[i-1]
        for k in range(1, max_k + 1):
            c_T = len(c_data)
            y_hat_Gaussian_Mix = mixture_exponentials(params_gaussian, c_T + k) + bias
            y_hat_SIR_Mix = mixture_SIR(params_SIR, c_T + k) + bias
            results_dict[country_name]['Week ' + str(k)]['GT'].append(data[i-1 + k])
            results_dict[country_name]['Week ' + str(k)]['Gaussian_mix'].append(y_hat_Gaussian_Mix[-1])
            results_dict[country_name]['Week ' + str(k)]['SIR_mix'].append(y_hat_SIR_Mix[-1])
    return results_dict

def visualize(results_dict, country_name, max_k):
    for k in range(1,max_k+1):
        gt = np.array(results_dict[country_name]['Week ' + str(k)]['GT'])
        predictions_Gaussian_Mix = np.array(results_dict[country_name]['Week ' + str(k)]['Gaussian_mix'])
        predictions_SIR_Mix = np.array(results_dict[country_name]['Week ' + str(k)]['SIR_mix'])
        print('MAPE Gaussian Mix')
        print(MAPE(gt, predictions_Gaussian_Mix))
        print('MAPE SIR Mix')
        print(MAPE(gt, predictions_SIR_Mix))
        plt.figure()
        plt.plot(gt)
        plt.plot(predictions_Gaussian_Mix)
        plt.plot(predictions_SIR_Mix)
        plt.legend(['GT', 'Gaussian_dict','SIR_dict', 'Gaussian_mix', 'SIR_mix','SLOW', 'SIR'])
        plt.show()
results_dict = main()
visualize(results_dict, 'Canada', 4)
# Needs fixing to work with smaller data sets
# finds the number of peaks and what the peaks are





