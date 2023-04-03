import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import SubpopulationsLib.DataProcessing as dp
import sklearn.metrics as metrics
import scipy as sp
import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from SubpopulationsLib.Subpopulations import mixture_exponentials, mixture_SIR, find_theta_sa
from SubpopulationsLib.Metrics import MAPE




#create a function to read the data from the csv file
def read_data(filename, start_date, end_date, country):
    S,I,R = dp.create_SIR_data(country, filename, './Data/UID_ISO_FIPS_LookUp_Table.csv', start_date, end_date)
    
    # indexes_weekly = np.arange(0,S.shape[0],7)

    # S = S[indexes_weekly]
    # I = I[indexes_weekly]
    # R = R[indexes_weekly]
    data = I[1:]
    return data

#function to get csv data for cases per country and date
def get_csv_data(start_date, end_date, country):
    df = pd.read_csv('./Data/WHO-COVID-19-global-data.csv')
    df = df[df['Country'] == country]
    df = df[df['Date_reported'] >= start_date]
    df = df[df['Date_reported'] <= end_date]
    
    return df

def get_no_recovery_date(country):
    df = pd.read_csv('./Data/time_series_covid19_recovered_global.csv')
    df = df[df['Country/Region'] == country]
    df = df.groupby("Country/Region").sum()
    date = date_to_datetime('9/1/20')
    while df[datetime_to_date(date)].values[0] >0:
        date += datetime.timedelta(days=1)
    #print recoveries of previous date
    return date

def datetime_to_date(date):
    date = date.strftime("%m/%d/%Y")
    if date[0] == "0":
        date = date[1:]
        if date [2] == "0":
            date = date[:2]+date[3:]
    if date[3] == "0":
        date = date[:3]+date[4:]
    
    date = date[ : -4] + date[-2:]
    return date

def date_to_datetime(date):
    date = date[0:-2]+"20"+date[-2:] #converts date to format that datetime can read
    date = datetime.datetime.strptime(date, "%m/%d/%Y")
    return date


#returns face covering and stay at home policies for a country given a start and end date as a tuple of np arrays
def get_policy_data(start, end, country):
    df = pd.read_csv('./Data/face-covering-policies-covid.csv')
    df = df[df['Entity'] == country]
    df = df[df['Date'] > start]
    df1 = df[df['Date'] < end]

    

    
    df = pd.read_csv('./Data/stay-at-home-covid.csv')
    df = df[df['Entity'] == country]
    df = df[df['Date'] > start]
    df2 = df[df['Date'] < end]

    df = pd.read_csv('./Data/school-closures-covid.csv')
    df = df[df['Entity'] == country]
    df = df[df['Date'] > start]
    df3 = df[df['Date'] < end]
    
    return (df1["facial_coverings"].to_numpy(),df2["stay_home_requirements"].to_numpy(),df3["school_closures"].to_numpy())

#create a function to plot the data using matplotlib
def plot_data(data, country,label):
    #create a figure

    plt.figure()
    #plot the data
    plt.plot(data, label = label)
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
    if tail_peak(data):
        next_params = get_next_params(params)
        params= np.insert(params, 1,next_params[0])
        params= np.insert(params, 3,next_params[1])
        params= np.insert(params, 5,next_params[2])

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
    sigma = params[1] -1 #assume that the next peak will have the same standard deviation as the previous peak
    coef = params[2]

    return mu, sigma, coef


def linear_regression():
    start_date = '2/28/20'
    end_date = '8/5/21'
    start = "2020-02-28"
    end = "2021-08-05"
    path = "./Data/"
    countries = ["Canada", "Australia","New Zealand", "Italy", "Sweden"]
    x = np.array([])
    y = np.array([])
    for country in countries:
        data = read_data(path, start_date, end_date, country)
        y = np.append(y, data)
        face_policy = get_policy_data(start, end, country)[0]
        home_policy = get_policy_data(start, end, country)[1]
        school_policy = get_policy_data(start, end, country)[2]
        

        x1 = face_policy.reshape(-1,1)
        x2 = home_policy.reshape(1,-1)
        x3 = school_policy.reshape(1,-1)
        
        #print(np.concatenate((x1,x2.T), axis = 1))

        if len(x)==0:
            x = (np.concatenate((x1,x2.T), axis = 1))
            x= np.concatenate((x,x3.T), axis = 1)
        else:
            temp = np.concatenate((x1,x2.T), axis = 1)
            temp = np.concatenate((temp,x3.T), axis = 1)
            x =np.append(x,temp,0)
    poly = PolynomialFeatures(degree =5)
    x = poly.fit_transform(x)
    p = np.random.permutation(len(x))
    x = x[p]
    y = y[p]
    size_x = len(x)
    size_y = len(y)
    x_train = x[:-size_x//5]
    x_test = x[-size_x//5:]
    y_train = y[:-size_y//5]
    y_test = y[-size_y//5:]
    

    # x_train = poly.fit_transform(x_train)
    # x_test = poly.fit_transform(x_test)

    model = LinearRegression().fit(x_train, y_train)
    print(model.score(x_test,y_test))
    print(model.score(x_train,y_train))

    model = LinearRegression().fit(x, y)
    print(model.score(x,y))
    
    

        
  



        
        
       
    




def main():
    path = "./Data/"
    start_date = '2/28/20'
    end_date = '2/28/21'
    # start_date = '2020-07-30'
    # end_date = '2021-03-30'
    countries = ["Canada", "Australia","New Zealand", "Italy", "Sweden"]

    for country in countries:
        
        data = read_data(path, start_date, end_date, country)
        plt.plot(data, label= country)
    plt.legend()
    plt.show()
    


    #policies = get_policy_data(start_date, end_date, country)

    #plt.plot(policies[0])
   # plt.plot(policies[1])

    # data2 = read_data(path, start_date, end_date2, country)

    # plot.plot(data2, color = 'green')
    # peaks, num_peaks = get_peaks(data)
    # #sir = fit_sir_model(data)
    # g_params = gaussian_params(data)
    
    # gaussian = forecast_gaussian(data, g_params,10)

    # plot.plot(gaussian, color = 'red')
    # #print("Gaussian MAPE: ", MAPE(data, gaussian))
    # #print("SIR MSE: ", mse(data, sir))    
 
    # # # print("SIR MAPE: ", MAPE(data, sir))    
    # #plot.plot(sir, color = 'green')
    # #plot.plot(peaks, data[peaks], 'bo')
    # plot.show()

linear_regression()

    
