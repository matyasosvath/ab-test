#!/usr/local/bin/python

from scipy.stats import t
from scipy.stats import sem
from numpy import mean
from numpy.random import randn
from numpy.random import seed
from math import sqrt


import numpy as np
import pandas as pd
from scipy import stats
import sys
import unittest


"""
4 moving parts
- sample size
- effect size
- power
- p-value

"""


alpha = 0.05
b = 1 - (float(alpha)/2)
confidence_level = stats.norm.ppf(b)
power = 0.8
#print(confidence_level) # 1.96

#n = sys.argv[2] # sample size



def read_data(data, needed_cols, **kwargs):
    df = pd.read_csv(data)
    df = df.iloc[:,int(needed_cols)]
    print(df)
    return df


class ABTest:
    def __init__(self, *data):
        self.data = np.array(data)
        print(self.data)
        print(f'Az adat típusa: {type(self.data)}')
        #self.n = int(float(n))  # sample size
        #print(f'Elemszám: {self.n}')

    def confidence_interval_proportion(self):
        """
        Calculates the confidence interval for the estimated proprtions (a point estimate).
        """
        prop = float(self.data[0])
        #print(f'Az adat típusa: {type(prop)}')
        #print(f'Adat: {prop}')
        prop_var = prop*(1-prop)
        lower_limit = np.round(prop - confidence_level*np.sqrt(prop_var/self.n), 7)
        upper_limit = np.round(prop + confidence_level*np.sqrt(prop_var/self.n), 7)

        return prop, (upper_limit, lower_limit)

    def confidence_interval_average(self):
        """
        Run like this: python3 abtest.py average 10.56 3 10
        """
        sample_mean = float(self.data[0])
        sample_std = float(self.data[1]) #sum(x-sample_mean for x in self.data)/self.n
        print(sample_mean, sample_std)

        se = sample_std/self.n  # Standard Error
        print(se)
        lower_limit = np.round(sample_mean - confidence_level*np.sqrt(se), 10)
        upper_limit = np.round(sample_mean + confidence_level*np.sqrt(se), 10)

        return avg, (upper_limit, lower_limit)

    def prop_converted():
        return pd.value_counts(self.data[0], normalize=True)

    def z_statistic_for_two_proportions(self):
        #p_a, p_b = self.prop_converted(self.data)
        p_a, p_b, n_a, n_b = self.data
        print(p_a, p_b)
        #n_a, n_b = self.data
        #n_a, n_b = len(n_a), len(n_b)
        print(f'N_a és N_b: {n_a}, {n_b}')

        p_a, p_b, n_a, n_b = float(p_a), float(p_b), float(n_a), float(n_b)

        prop_a = p_a/n_a
        prop_b = p_b/n_b
        # se_pool = (1/n_a+1/n_b)

        p = (p_a + p_b)/(n_b + n_a)
        #print(p)
        z_score = (prop_b - prop_a) / np.sqrt(p*(1-p)*((1/n_a)+(1/n_b)))

        pvalue = stats.norm.pdf(abs(z_score)) * 2 # two-tailed
        return (z_score, pvalue) # abs(z_score) ???

    def t_statistic_for_two_averages(self):
        avg_1, avg_2, std_1, std_2, n_1, n_2 = self.data
        avg_1, avg_2, std_1, std_2, n_1, n_2 = float(avg_1), float(avg_2), float(std_1), float(std_2), float(n_1), float(n_2)  
        
        # print(x, y)


        # std1, std2 = x.std(), y.std()
        # # print(f'standard devs {std1},{std2}')
        # se1 = std1/np.sqrt(n1)
        # se2 = std2/np.sqrt(n2)
        # print(f'standard errors {se1},{se2}')

        # se = np.sqrt(se1**2 + se2**2)
        # t_stat = (x.mean() - y.mean()) / se

        # df = n1 + n2 - 2  # degrees of freedom

        # p = (1 - stats.t.cdf(abs(t_stat), df)) * 2  # two-tailed
        # print(p)
        # # p = stats.t.sf(t_stat, df)
        # # print(p)
        # return t_stat, p


        se1 = std_1/np.sqrt(n_1)
        se2 = std_2/np.sqrt(n_2)
        # print(f'standard errors {se1},{se2}')

        #se = np.sqrt(se1**2 + se2**2)
        # calculate standard errors
        #se1, se2 = sem(x), sem(y)
        
        # standard error on the difference between the samples
        sed = sqrt(se1**2.0 + se2**2.0)
        
        # t-statistic
        t_stat = (avg_1 - avg_2) / sed
        
        # degrees of freedom
        df = n_1 + n_2 - 2
        
        # P-value
        p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
        # return everything
        return t_stat, p


#TODO: tests

if __name__ == '__main__':
    # alpha = sys.argv[1]
    # sample_size = sys.argv[2]
    # effect_size = sys.argv[3]
    # power = sys.argv[4]

    # b = 1 - (float(alpha)/2)
    # confidence_level = stats.norm.ppf(b)

  

    import sys


    df = sys.argv[1]
    cols = sys.argv[2]

    print(read_data(df, cols))

    if sys.argv[1] == 'proportion':
        p = sys.argv[2]
        n = sys.argv[3]

        ab = ABTest(p)
        p, ci = ab.confidence_interval_proportion()
        print(f'Proportions was {p} with CI 95% {ci}.')
        print('\n')



    if sys.argv[1] == 'proportions':

        prop_1 = sys.argv[2]
        prop_2 = sys.argv[3]
        n_a = sys.argv[4]
        n_b = sys.argv[5]

        ab = ABTest(prop_1,prop_2, n_a, n_b)
        print(f'{ab.z_statistic_for_two_proportions()}')


    if sys.argv[1] == 'average':

        avg = sys.argv[2]
        std = sys.argv[3]
        n = sys.argv[4]
        
        ab = ABTest(avg, std)
        avg, ci = ab.confidence_interval_average()
        print(f'Average was {avg} with CI 95% {ci}.')
        print('\n')


    if sys.argv[1] == 'averages':

        avg_1 = sys.argv[2]
        avg_2 = sys.argv[3]
        
        std_1 = sys.argv[4]
        std_2 = sys.argv[5]
        
        n_1 = sys.argv[6]
        n_2 = sys.argv[7]
        
        ab = ABTest(avg_1, avg_2, std_1, std_2, n_1, n_2)
        t, p = ab.t_statistic_for_two_averages()
        print(f'Test statistic is {t} and p-value is {p}')



    
    # x = pd.Series(np.random.randint(10, 20, size=30))
    # y = pd.Series(np.random.randint(15, 30, size=30))
    # print(x.mean(), y.mean())

    # ab = ABTest(x, y)
    # print(ab.t_statistic_for_two_averages())
    # print(stats.ttest_ind(x, y, equal_var=False))
    # print(stats.ttest_ind(x, y, equal_var=True))
