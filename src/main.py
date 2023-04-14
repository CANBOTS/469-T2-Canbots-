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

from sklearn.model_selection import train_test_split

from SubpopulationsLib.Subpopulations import mixture_exponentials, mixture_SIR, find_theta_sa

from SubpopulationsLib.Metrics import MAPE

#function to get the weekly infections for a coutnrty given a start and end date
def read_data(filename, start_date, end_date, country):
    S,I,R = dp.create_SIR_data(country, filename, './Data/UID_ISO_FIPS_LookUp_Table.csv', start_date, end_date)
    
    indexes_weekly = np.arange(0,S.shape[0],7)
    

    S = S[indexes_weekly]
    I = I[indexes_weekly]
    R = R[indexes_weekly]
    data = I
    return data

#function to get daily infeted cases, used for the regression model
def data_daily(filename, start_date, end_date, country):
    S,I,R = dp.create_SIR_data(country, filename, './Data/UID_ISO_FIPS_LookUp_Table.csv', start_date, end_date)
    data = I
    return data

#function to get csv data for cases per country and date, Not being used 
def get_csv_data(country,start_date, end_date):
    df = pd.read_csv('./Data/WHO-COVID-19-global-data.csv')
    df = df[df['Country'] == country]
    df = df[df['Date_reported'] > start_date]
    df = df[df['Date_reported'] < end_date]

    cases = df["New_cases"].to_numpy()
    cases = sp.signal.savgol_filter(cases, len(cases)//2,7)
    
    return cases

#returns the last day that there were no recoveries reported for a country
def get_no_recovery_date(country):
    df = pd.read_csv('./Data/time_series_covid19_recovered_global.csv')
    df = df[df['Country/Region'] == country]
    df = df.groupby("Country/Region").sum()
    date = date_to_datetime('9/1/20')
    while df[datetime_to_date(date)].values[0] >0:
        date += datetime.timedelta(days=1)
    #print recoveries of previous date
    return date

#returns population density of a country
def population_density(country):
    df = pd.read_csv('./Data/population_density.csv')
    df = df[df['Country Name'] == country]
    return df["2020"].values[0]

#returns urban population proportion of a country
def urban(country):
    df = pd.read_csv('./Data/urban_population.csv')
    df = df[df['Country Name'] == country]
    return df["2020"].values[0]

#returns population of a country
def population(country):
    df = pd.read_csv('./Data/world_population.csv')
    df = df[df['Country/Territory'] == country]
    return df["2020 Population"].values[0]

#converts datetime to date for the csv files
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


#converts date to datetime for the csv files
def date_to_datetime(date):
    date = date[0:-2]+"20"+date[-2:] #converts date to format that datetime can read
    date = datetime.datetime.strptime(date, "%m/%d/%Y")
    return date


#returns face covering and stay at home policies for a country given a start and end date as a tuple of np arrays
def get_policy_data(start, end, country):
    df = pd.read_csv('./Data/face-covering-policies-covid.csv')
    df = df[df['Entity'] == country]
    df = df[df['Date'] >= start]
    df1 = df[df['Date'] < end]

    df = pd.read_csv('./Data/stay-at-home-covid.csv')
    df = df[df['Entity'] == country]
    df = df[df['Date'] >= start]
    df2 = df[df['Date'] < end]

    df = pd.read_csv('./Data/school-closures-covid.csv')
    df = df[df['Entity'] == country]
    df = df[df['Date'] >= start]
    df3 = df[df['Date'] < end]
    
    return (df1["facial_coverings"].to_numpy(),df2["stay_home_requirements"].to_numpy(),df3["school_closures"].to_numpy())

#create a function to plot the data using matplotlib, kinda unnecessary tbh
def plot_data(data, country,label):
    #create a figure

    plt.figure()
    #plot the data
    plt.plot(data, label = label)
    plt.xlabel("Time (weeks)")
    plt.ylabel("Number of Infected people")
    plt.title("Number of Infected people over time in " + country)
    return plt

#method which returns the peaks and number of peaks for the data
def get_peaks(data):
    smoothed_data =  sp.signal.savgol_filter(data, len(data)//2,7)
    peaks = sp.signal.find_peaks(smoothed_data)[0]

    if len(peaks) == 0:
        return 0,2

    return peaks, len(peaks)

#fits the sir model but I havent changed this from robertos code
def fit_sir_model(data):
    num_mixtures = get_peaks(data)[1]
    bound_S = (0,1E8)
    bound_beta = (0,1)
    bound_gamma = (0,1)
    bound_coef = (0,1000)
    bound_k = (0,40)
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
    T = len(norm_I)+4
    y_hat_gaussian = mixture_exponentials(params, T) + bias
    
    return y_hat_gaussian

#fits the gaussians on a given data set and returns the parameters
def gaussian_params(data):
    num_mixtures = get_peaks(data)[1]
    bounds_mu = (0,40)
    bounds_sigma = (1,6)
    bounds_coef = (0,1500000)

    bound_list_Gaussian = [bounds_mu, bounds_sigma, bounds_coef]

    bounds_Gaussian = list()

    for element in bound_list_Gaussian:
        for i in range(num_mixtures):
            bounds_Gaussian.append(element)
    bias = np.min(data)
    norm_I = data - bias

    params_gaussian = find_theta_sa(bounds_Gaussian, norm_I, mixture_exponentials) 
    return params_gaussian

#forecasts the gaussian for a given number of weeks and returns the entire data
def forecast_gaussian( data, params, steps, start_date, country,model):
    bias = np.min(data)
    norm_I = data - bias
    T = len(norm_I)
    #print(params)
    if tail_peak(data):
        next_params = get_next_params(start_date, params, country,model)

        if next_params:
            if len(params) == 3: #one previous gaussian
                params= np.insert(params, 1,next_params[0])
                params= np.insert(params, 3,next_params[1])
                params= np.insert(params, 5,next_params[2])
            elif len(params) == 6: #two previous gaussians
                params= np.insert(params, 2,next_params[0])
                params= np.insert(params, 5,next_params[1])
                params= np.insert(params, 8,next_params[2])
            elif len(params) == 9: #three previous gaussians
                params= np.insert(params, 3,next_params[0])
                params= np.insert(params, 7,next_params[1])
                params= np.insert(params, 11,next_params[2])
    y_hat_gaussian = mixture_exponentials(params, T) + bias
    # plot_gaussians(params, data)
    for i in range(steps):
        c_T = len(y_hat_gaussian)
        new = mixture_exponentials(params, c_T + i) + bias

        y_hat_gaussian = np.append(y_hat_gaussian,[new[-1]])
    
    return y_hat_gaussian
    

#plots all the individual gaussians
def plot_gaussians(params,data):
    adder = len(params)//3
    for i in range(adder):
        new_params = (params[i], params[i+adder], params[i+adder*2])
        #print(new_params)
        plt.plot(fit_gaussian_model(data, new_params), label = "Gaussian " + str(i+1))

def is_holdiday(date, country):
    #get the holidays for the country
    holidays = get_holidays(country)
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
def get_next_params(start_date, params, country,model):

    if len(params) ==6:
        means = [params[0],params[1]]
        i = means.index(max(means))
        params = (params[i],params[i+2],params[i+4])
    elif len(params) ==9:
        means = [params[0],params[1],params[2]]
        i = means.index(max(means))
        params = (params[i],params[i+3],params[i+6])
    
    #Just so it doesnt fit a gaussian too far into the future, thends to mess with predictions 
    if 45<params[0]<51:
        return False
    mu = params[0] + 16 #assume that the next peak will be 18 weeks after the previous peak
    

    sigma = params[1]*0.7 #assume that the next peak will have the same standard deviation as the previous peak

    '''
    Use below code to use the regression model to get the next coefficient
    '''
    #convert start_date to a datetime object
    # start_date = date_to_datetime(start_date)
    # mu_date = start_date + datetime.timedelta(days=mu*7) #convert the number of weeks to a date
    # modl = model
    # feautures = get_feautures(mu_date, country)
    # coef = modl.predict(feautures)

    '''
    Use below code to use the previous coefficient to get the next coefficient
    '''
    if len(params)>0:
        coef = params[2]

    return mu,sigma,coef


#Function to get the regression feuatures for a given date and country
def get_feautures(date, country):
    #format the date as a datetime object
    #date = datetime.datetime.strptime(date, "%Y-%m-%d")
    start = date-datetime.timedelta(days=1) 
    #format the date as a string
    start = start.strftime("%Y-%m-%d")

    end = date+datetime.timedelta(days=1)
    end = end.strftime("%Y-%m-%d")
    
    features = get_policy_data(start,end, country)
    x = np.array([])
    for feature in features:
        x = np.append(x,feature[0])
    
    x = np.append(x,population_density(country))
    x = np.append(x, urban(country))
    #print(x)
    poly = PolynomialFeatures(degree =5)
    
    x = poly.fit_transform([x])
    return x
    
#function to get the the training daata for the regression model
def get_x_y():
    start_date = '2/28/20'
    end_date = '8/5/21'
    start = "2020-02-28"
    end = "2021-08-05"
    path = "./Data/"
    countries = ["Canada", "Australia","New Zealand", "Italy", "Sweden", "United Kingdom","China","South Korea","Japan"]
    x = np.array([])
    y = np.array([])
    for country in countries:
        data = data_daily(path, start_date, end_date, country)
        y = np.append(y, data)
        
        face_policy = get_policy_data(start, end, country)[0]
        home_policy = get_policy_data(start, end, country)[1]
        school_policy = get_policy_data(start, end, country)[2]
        pop = population_density(country)*np.ones(len(data))
        urban_pop = urban(country)*np.ones(len(data))

        x1 = face_policy.reshape(-1,1)
        x2 = home_policy.reshape(-1,1)
        x3 = school_policy.reshape(-1,1)
        x4 = pop.reshape(-1,1)
        x5 = urban_pop.reshape(-1,1)
        


        if len(x)==0:
            x = (np.concatenate((x1,x2), axis = 1))
            x= np.concatenate((x,x3), axis = 1)
            x = np.concatenate((x,x4), axis = 1)
            x = np.concatenate((x,x5), axis = 1)
            
        else:
            temp = np.concatenate((x1,x2), axis = 1)
            temp = np.concatenate((temp,x3), axis = 1)
            temp = np.concatenate((temp,x4), axis = 1)
            temp = np.concatenate((temp,x5), axis = 1)
            x =np.append(x,temp,0)
    
    return x,y




def linear_regression():
    
    x,y = get_x_y()
  
    #fits the input data to a fifth degree polynomial
    poly = PolynomialFeatures(degree =5)
    x = poly.fit_transform(x)
    
    #splits the data into training and testing data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


    model = LinearRegression().fit(x_train, y_train)
    print(model.score(x_test,y_test))
    model = LinearRegression().fit(x, y)
    print(model.score(x,y))
    return model


    
    



def main():
    path = "./Data/"
    # start_date = '3/30/20'
    # end_date = '1/30/21'
    # end_date2 = '3/30/21'
    # # start_date = '2020-02-27'
    # # end_date = '2021-08-05'
    # country = 'Italy'
    start_date = '7/30/20'
    end_date = '4/20/21'
    end_date2 = '5/20/21'
    countries = ["Canada", "Australia", "Sweden", "United Kingdom","China","South Korea","Japan","Germany","Singapore"]
    #country = "Mexico"
    
    


    #policies = get_policy_data(start_date, end_date, country)

    #plt.plot(policies[0])
   # plt.plot(policies[1])
    for_me = {}
    model = 3
    heights = {}
    errors ={}
    for country in countries:
        data = read_data(path, start_date, end_date, country)

        data2 = read_data(path, start_date, end_date2, country)
        mapes = []
        for i in range(30):
            print(country,i)
            g_params = gaussian_params(data)
            gaussian = forecast_gaussian(data, g_params,4,start_date, country, model)
            mape = MAPE(data2[-4:], gaussian[-4:])[0]
            mapes.append(mape)
        interval = sp.stats.t.interval(confidence=0.95,
              df=len(mapes)-1,
              loc=np.mean(mapes), 
              scale=sp.stats.sem(mapes))
        heights[country] = interval[1]-interval[0]
        errors[country] = np.mean(mapes)
        for_me[country] = mapes
    plt.bar(errors.keys(), errors.values(), yerr = heights.values(), color='g')
    plt.title("MAPE assuming previous magnitude")
    plt.ylabel("MAPE")
    plt.xlabel("Country")
    plt.ylim((0,100))
    print(for_me)
    #plot = plot_data(data2, country,"data")
    # plt.plot(data, color = 'blue')
    #plt.plot(data2, color = 'green')
    peaks, num_peaks = get_peaks(data)
    #print(num_peaks)
    #sir = fit_sir_model(data)
    
    
    
    #gaussian = fit_gaussian_model(data, g_params)

    #plt.plot(gaussian, color = 'red')
    # plt.legend()
    plt.show()
    
    #print("Gaussian MAPE: ", MAPE(data2, gaussian))
    # #print("SIR MSE: ", mse(data, sir))    
 
    # # # print("SIR MAPE: ", MAPE(data, sir))    
    # #plot.plot(sir, color = 'green')
    #plot.plot(peaks, data[peaks], 'bo')

def main2():
    path = "./Data/"
    # start_date = '3/30/20'
    # end_date = '1/30/21'
    # end_date2 = '3/30/21'
    # # start_date = '2020-02-27'
    # # end_date = '2021-08-05'
    # country = 'Italy'
    start_date = '7/30/20'
    end_date = '4/20/21'
    end_date2 = '5/20/21'
    country = "Singapore"
    
    


    #policies = get_policy_data(start_date, end_date, country)

    #plt.plot(policies[0])
   # plt.plot(policies[1])
    model = linear_regression()
    
    data = read_data(path, start_date, end_date, country)

    data2 = read_data(path, start_date, end_date2, country)
       
    g_params = gaussian_params(data)
    gaussian = forecast_gaussian(data, g_params,4,start_date, country, model)

    plt.plot(data, color = 'blue')
    plt.plot(data2, color = 'green')
    peaks, num_peaks = get_peaks(data)
    print(num_peaks)
    #sir = fit_sir_model(data)
    
    
    
    #gaussian = fit_gaussian_model(data, g_params)

    plt.plot(gaussian, color = 'red')
    # plt.legend()
    plt.show()
    
    print("Gaussian MAPE: ", MAPE(data2, gaussian))
    # #print("SIR MSE: ", mse(data, sir))    
 
    # # # print("SIR MAPE: ", MAPE(data, sir))    
    # #plot.plot(sir, color = 'green')
    #plot.plot(peaks, data[peaks], 'bo')
def graphics():
    path = "./Data/"
   
    start_date = '7/30/20'
    end_date = '4/15/21'
    country = "Japan"
    data = read_data(path, start_date, end_date, country)
    smoothed_data =  sp.signal.savgol_filter(data, len(data)//2,5)
    plt.plot(data, color = 'blue',label = "Unsmoothed Data")
    plt.plot(smoothed_data, color = 'red', label = "Smoothed Data")
    peaks, num_peaks = get_peaks(data)

    plt.plot(peaks, data[peaks], 'bo', color = "green")

    plt.legend()
    plt.show()




#graphics()
# print("regression")
linear_regression()
#main()
#print(population("US"))
# print("neural")
# neural_net()
#print(get_csv_data("Canada", "2020-02-28", "2021-08-05"))
#print(get_feautures("2020-06-28", "Canada"))

    
