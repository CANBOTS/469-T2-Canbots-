# This is a different version of main, to implement the code that Roberto wrote

from SubpopulationsLib import DataProcessing, InfectiousModels, Metrics, Subpopulations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint




# SfIR model
class SfIR:

    def __init__(self):
        self.model = None
        self.N = None

    def train(self, S, I, R, beta, gamma):
        self.N = len(S)

        dSdt = -beta * S * I / self.N
        dIdt = beta * S * I / self.N - gamma * I
        dRdt = gamma * I
        dBdt = -2 * I / self.N * (np.mean(beta) - beta[0]) + 2 * mu * (np.mean(beta) - beta[0]) + 2 * D * beta[1]

        return dSdt, dIdt, dRdt, dBdt

    def get_train(self, S, I, R, beta, gamma):
        y0 = S0, I0, R0, B0 = S[0], I[0], R[0], beta[0]
        return np.array([dSdt, dIdt, dRdt, dBdt])


def SIR(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def SfIR(y, t, N, beta0, beta1, gamma, mu, D):
    S, I, R, beta = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    dBdt = -2 * I / N * (beta - beta0)**2 + 2 * mu * (beta - beta0) + 2 * D * beta1
    return dSdt, dIdt, dRdt, dBdt


def split_SIR(S,I,R, t_train, t_test):
    S_train = S[:t_train]
    I_train = I[:t_train]
    R_train = R[:t_train]

    S_test = S[t_train:]
    I_test = I[t_train:]
    R_test = R[t_train:]

    time_train = np.linspace(0, t_train, t_train + 1)
    time_test = np.linspace(t_train + 1, t_train + t_test, t_test)

    return S_train, I_train, R_train, S_test, I_test, R_test, time_train, time_test


def main():

    # Paths
    file_path = 'data/covid_data/'
    lookup_table_path = "data/UID_ISO_FIPS_LookUp_Table.csv"

    # Vars for testing
    country_name = "Canada"
    start_date = "1/22/20"
    end_date = "1/22/22"

    # Initial conditions
    beta_mean = 0.2
    beta0 = 0.1
    beta1 = 0.3
    gamma = 0.1
    
    # Get the data as SIR
    S,I,R,N = DataProcessing.create_SIR_data(country_name, file_path, lookup_table_path, start_date, end_date)

    # Split the data
    t_train = int(len(S) * 0.8)
    t_test = len(S) - t_train

    S_train, I_train, R_train, S_test, I_test, R_test, time_train, time_test  = split_SIR(S,I,R, t_train, t_test)

    #sol = odeint(SIR, [S[0],I[0],R[0]], time_train, args=(N, beta0, gamma))
    #forecast = odeint(SIR, sol[-1], time_test, args=(N, beta0, gamma))

    sol = odeint(SfIR, [S_train[0],I_train[0], R_train[0], beta_mean], time_train, args=(N, beta0, beta1, gamma, 0.1, 0.1))
    #forecast = odeint(SfIR, sol[-1], time_test, args=(N, beta0, beta1, gamma, 0.1, 0.1))

    # Plot the data
    # Plot the results
    plt.plot(time_train, sol[:,0], label='Susceptible')
    plt.plot(time_train, sol[:,1], label='Infected')
    plt.plot(time_train, sol[:,2], label='Recovered')
    

    #plt.plot(t, S, label='Susceptible')
    #plt.plot(t, I, label='Infected')
    #plt.plot(t, R, label='Recovered')
    
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SIR Model')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
