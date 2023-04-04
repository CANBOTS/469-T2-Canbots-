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

    # Convert start and end dates to strings
    start = start.strftime("%m/%d/%Y")
    end = end.strftime("%m/%d/%Y")
    
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


def forecast_sir_model(data, params, steps):

    bias = np.min(data)
    norm_I = data - bias
    T = len(norm_I)

    if tail_peak(data):
        next_params = get_next_params(params)
        params= np.insert(params, 1,next_params[0])
        params= np.insert(params, 3,next_params[1])
        params= np.insert(params, 5,next_params[2])

    y_hat_sir = mixture_SIR(params, T) + bias
    
    for i in range(steps):
        c_T = len(y_hat_sir)
        new = mixture_SIR(params, c_T + i) + bias

        y_hat_sir = np.append(y_hat_sir,[new[-1]])
    
    return y_hat_sir


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

    holiday_data = pd.read_csv("holiday_calendar.csv")

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


def make_features(data, country, dates):
    
    # Get the start and end dates from the data
    start_date = dates[0]
    end_date = dates[-1]

    # Get the policy data
    face_policy = get_policy_data(start_date, end_date, country)[0].reshape(1,-1)
    home_policy = get_policy_data(start_date, end_date, country)[1].reshape(1,-1)
    school_policy = get_policy_data(start_date, end_date, country)[2].reshape(1,-1)

    # Get the days from holidays as 2 arrays
    prev_holiday = np.array([days_from_holiday(date,country)[0] for date in dates]).reshape(1,-1)
    next_holiday = np.array([days_from_holiday(date,country)[1] for date in dates]).reshape(1,-1)
    
    # Combine the features
    features = np.concatenate((face_policy, home_policy, school_policy, prev_holiday, next_holiday), axis=1)

    return features



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


# Function to return the number of days from the previous holiday given a date and country
def days_from_holiday(date,country):
    
    # Read in holiday data
    holiday_data = pd.read_csv("./Data/holiday_calendar.csv")

    # Get holiday data for country
    holidays = holiday_data[holiday_data["Country"] == country.lower()]["Date"].values

    # Strip year from date
    #date = date.strftime("%m-%d")

    # Convert date to string with year
    date = date.strftime("%Y-%m-%d")

    # Strip year from holidays
    holidays = [holiday[5:] for holiday in holidays]

    # Add years from 2020 to 2023 to holidays
    holidays = [f"{year}-{holiday}" for year in range(2020,2024) for holiday in holidays]

    # Get holiday before date
    prev_holiday = max([holiday for holiday in holidays if holiday <= date])

    # Get holiday after date
    next_holiday = min([holiday for holiday in holidays if holiday >= date])

    # Fix format
    prev_holiday = datetime.datetime.strptime(prev_holiday, "%Y-%m-%d")
    next_holiday = datetime.datetime.strptime(next_holiday, "%Y-%m-%d")

    try:
        date = datetime.datetime.strptime(date, "%Y-%m-%d", errors="ignore")

        # Get number of days from previous holiday
        days_from_prev_holiday = (date - prev_holiday).days

        # Get number of days to next holiday
        days_to_next_holiday = (next_holiday - date).days

        return days_from_prev_holiday, days_to_next_holiday
    
    except:
        # This happens on Feb 28th
        # Get year from date
        year = date[:4]

        if year + "-02-28" in holidays:
            return 0, 0

        else:
            date = datetime.datetime.strptime(f"{year}-02-27", "%Y-%m-%d")
            
            # Get number of days from previous holiday
            days_from_prev_holiday = (date - prev_holiday).days

            # Get number of days to next holiday
            days_to_next_holiday = (next_holiday - date).days

            return days_from_prev_holiday + 1, days_to_next_holiday - 1





class GaussianMixtureModel:

    def __init__(self, bounds_mu=(0,50), bounds_sigma=(1,6), bounds_coef=(0,300000)):
        
        self.bounds_mu = bounds_mu
        self.bounds_sigma = bounds_sigma
        self.bounds_coef = bounds_coef
        self.bounds_list = [bounds_mu, bounds_sigma, bounds_coef]


    def fit(self, data, params):
        
        bias = np.min(data)
        norm_I = data - bias
        
        T = len(norm_I)
        y_hat_gaussian = mixture_exponentials(params, T) + bias
    
        return y_hat_gaussian


    def get_params(self, data):
        
        bounds_Gaussian = list()

        for element in self.bounds_list:
            for i in range(get_peaks(data)[1]):
                bounds_Gaussian.append(element)
    
        bias = np.min(data)
        norm_I = data - bias

        params_gaussian = find_theta_sa(bounds_Gaussian, norm_I, mixture_exponentials) 
        return params_gaussian


    def get_next_params(self, params):
        mu = params[0] + 18
        sigma = params[1] -1
        coef = params[2]

        return mu, sigma, coef


    def forecast(self, data, params, steps):
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


    def linreg(self,features):
        pass

    
def split_data(data, train_size=0.8):
    
    # Split into train and test data
    train_data = data[:int(len(data)*train_size)]
    test_data = data[int(len(data)*train_size):]

    return train_data, test_data


def get_dates(start_date, end_date, train_size=0.8):
    
    # Get range of dates
    dates = pd.date_range(start_date, end_date)

    # Split into train and test dates
    train_dates = dates[:int(len(dates)*train_size)]
    test_dates = dates[int(len(dates)*train_size):]

    return train_dates, test_dates

class SIR_multi:

    def __init__(self, n_subpops, beta, gamma, S0, I0, R0):
        self.n_subpops = n_subpops
        self.beta = beta
        self.gamma = gamma
        self.S = S0
        self.I = I0
        self.R = R0

    def step(self):
        new_I = np.zeros(self.n_subpops)
        new_R = np.zeros(self.n_subpops)
        for i in range(self.n_subpops):
            new_I[i] = self.beta[i] * np.dot(self.S, self.I[i])
            new_R[i] = self.gamma[i] * self.I[i]

        for i in range(self.n_subpops):
            self.S[i] -= new_I[i]
            self.I[i] += new_I[i] - new_R[i]
            self.R[i] += new_R[i]


    def simulate(self, num_steps, data=None):
        S = [self.S]
        I = [self.I]
        R = [self.R]
        if data is None:
            data = np.zeros((self.n_subpops,num_steps))
            data[:,0] = self.I

        for i in range(num_steps):
            self.I = data[:,i]
            self.step()
            S.append(self.S)
            I.append(self.I)
            R.append(self.R)

        return np.array(S), np.array(I), np.array(R)



class SfIR:

    def __init__(self, S0, f0, I0, R0, H0, D0, N0, mu, nu, gamma, alpha, delta, epsilon, rho, kappa, D):
        # Set initial conditions
        self.S = S0
        self.f = f0
        self.I = I0
        self.R = R0
        self.H = H0
        self.D = D0
        self.N = N0
        self.mu = mu
        self.nu = nu
        self.gamma = gamma
        self.alpha = alpha
        self.delta = delta
        self.epsilon = epsilon
        self.rho = rho
        self.kappa = kappa
        self.D = D
        self.num_subpops = D.shape[0]


    def step(self, dt):
        p = np.random.uniform(size = self.num_subpops)
        SfI = self.D @ np.array([self.S, self.f, self.I])
        fI = np.sum(SfI[1:])
        V = self.kappa * (self.N - np.sum(SfI))

        dSdt = self.mu * (self.N - self.S) - (1-p) * SfI[0] * fI / self.N - self.rho * self.S
        dfdt = (1- self.mu) * V - self.kappa * self.f - self.epsilon * self.f
        dIdt = (1 - self.alpha) * self.gamma * fI - (1 - self.alpha) * self.delta * self.I - self.alpha * self.H
        dRdt = self.gamma * (1 - self.alpha) * self.I
        dHdt = self.alpha * self.gamma * self.I - self.epsilon * self.H
        dDdt = self.rho * self.S + self.delta * (1 - self.alpha) * self.I + self.epsilon * self.f + self.epsilon * self.H

        self.S += dSdt * dt
        self.f += dfdt * dt
        self.I += dIdt * dt
        self.R += dRdt * dt
        self.H += dHdt * dt
        self.D += dDdt * dt

    def run(self, t_end, dt):
        ts = [0]
        Ss = [self.S]
        fs = [self.f]
        Is = [self.I]
        Rs = [self.R]
        Hs = [self.H]
        Ds = [self.D]

        while ts[-1] < t_end:
            self.step(dt)

            ts.append(ts[-1] + dt)
            Ss.append(self.S)
            fs.append(self.f)
            Is.append(self.I)
            Rs.append(self.R)
            Hs.append(self.H)
            Ds.append(self.D)

        return ts, Ss, fs, Is, Rs, Hs, Ds


    @staticmethod
    def from_data(data, N, mu, nu, gamma, alpha, delta, epsilon, rho, kappa, D):
        ts, Is, Rs, Hs, Ds = data

        I0 = Is[0]
        R0 = Rs[0]
        H0 = Hs[0]
        D0 = Ds[0]
        S0 = N - I0 - R0 - H0 - D0
        f0 = 0

        SIR = S0 + I0 + R0
        SIRH = SIR + H0
        SIRHD = SIRH + D0

        f_star = (SIRH - SIR * np.exp(-kappa * ts[-1])) / (SIRHD - SIRH * np.exp(-kappa * ts[-1]))

        f0 = np.array([f_star] * D.shape[0])

        return SfIR(S0, f0, I0, R0, H0, D0, N, mu, nu, gamma, alpha, delta, epsilon, rho, kappa, D)

    def simulate(self, t):

        Is = np.zeros(len(t))
        Rs = np.zeros(len(t))
        Hs = np.zeros(len(t))
        Ds = np.zeros(len(t))

        Is[0] = self.I
        Rs[0] = self.R
        Hs[0] = self.H
        Ds[0] = self.D
        
        for i in range(1, len(t)):
            dt = t[i] - t[i-1]

            dSdt = -self.mu * self.S
            dfdt = self.mu * self.S - self.nu * self.f
            dIdt = (1 - self.alpha) * self.rho * self.f - (self.gamma + self.kappa + self.epsilon) * self.I
            dRdt = self.gamma * self.I
            dHdt = self.alpha * self.rho * self.f - (self.delta * self.H)
            dDdt = self.epsilon * self.I + self.delta * self.H

            self.S += dSdt * dt
            self.f += dfdt * dt
            self.I += dIdt * dt
            self.R += dRdt * dt
            self.H += dHdt * dt
            self.D += dDdt * dt


            self.S = max(self.S, 0)
            self.f = max(self.f, 0)
            self.I = max(self.I, 0)
            self.R = max(self.R, 0)
            self.H = max(self.H, 0)
            self.D = max(self.D, 0)

            Is[i] = self.I
            Rs[i] = self.R
            Hs[i] = self.H
            Ds[i] = self.D

            return np.column_stack((t, Is, Rs, Hs, Ds))



def main():

    N = 1000000
    mu = 0.0001
    nu = 0.0001
    gamma = 0.1
    alpha = 0.1
    delta = 0.1
    epsilon = 0.1
    rho = 0.1
    kappa = 0.1
    D = np.array([0.1, 0.1, 0.1, 0.1, 0.1])

    I0 = 100
    R0 = 0
    H0 = 0
    D0 = 0
    S0 = N - I0 - R0 - H0 - D0
    f0 = 0

    t = np.linspace(0, 100, 1000)
    data = [t, np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t))]

    model = SfIR(S0, f0, I0, R0, H0, D0, N, mu, nu, gamma, alpha, delta, epsilon, rho, kappa, D)

    results = model.simulate(t)
    Is, Rs, Hs, Dcs = results[:,0], results[:,1], results[:,2], results[:,3]

    plt.plot(t, Is, label='I')
    plt.plot(t, Rs, label='R')
    plt.plot(t, Hs, label='H')
    plt.plot(t, Dcs, label='D')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()



    

