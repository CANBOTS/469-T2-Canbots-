

import csv
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import SubpopulationsLib.DataProcessing as dp



#create a function to read the data from the csv file
def read_data(filename, start_date, end_date, country):
    S,I,R = dp.create_SIR_data(country, filename, '../Data/UID_ISO_FIPS_LookUp_Table.csv', start_date, end_date)
    
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

    plt.show()

def main():
    path = "../Data/"
    start_date = '1/22/20'
    end_date = '1/22/21'
    country = 'Canada'
    data = read_data(path, start_date, end_date, country)
    plot_data(data, country)


    
main()