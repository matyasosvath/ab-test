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

# b = 1 - (float(alpha)/2)
# confidence_level = stats.norm.ppf(b)


class ABTest:
    def __init__(self, *data):
        self.data = np.array(data)
        print(self.data)
        self.n = len(data)  # sample size

    def confidence_interval_proportion(self, prop):
        """
        prop/p_hat = point estimate
        n: sample size
        """

        prop_var = prop*(1-prop)
        prop_neg = prop - confidence_level*np.sqrt(prop_var/self.n)
        prop_pos = prop + confidence_level*np.sqrt(prop_var/self.n)

        return prop, (prop_neg, prop_pos)

    def confidence_interval_average(self):
        sample_mean = sum(self.data)/len(self.data)
        sample_var = sum(x-sample_mean for x in self.data)/self.n
        print(sample_mean, sample_var)

        se = sample_var/self.n  # Standard Error
        avg_minus = sample_mean - confidence_level*np.sqrt(se)
        avg_plus = sample_mean + confidence_level*np.sqrt(se)

        return (avg_minus, avg_plus)

    def prop_converted():
        return pd.value_counts(self.data[0], normalize=True)

    def z_statistic_for_two_proportions(self):
        p_a, p_b = self.prop_converted(self.data)
        print(p_a, p_b)
        n_a, n_b = self.data
        n_a, n_b = len(n_a), len(n_b)
        print(f'N_a és N_b: {n_a}, {n_b}')

        # se_pool = (1/n_a+1/n_b)

        p = (p_a + p_b)/(n_b + n_a)
        z_score = (p_b - p_a) / np.sqrt(p*(1-p)(1/n_a+1/n_b))

        pvalue = stats.norm.pdf(z_score) * 2
        return (z_score, pvalue)

    def t_statistic_for_two_averages(self):
        x, y = self.data
        # print(x, y)

        n1, n2 = len(x), len(y)
        # print(f'adatok {n1},{n2}')

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
        mean1, mean2 = mean(x), mean(y)

        se1 = x.std()/np.sqrt(n1)
        se2 = y.std()/np.sqrt(n2)
        # print(f'standard errors {se1},{se2}')

        #se = np.sqrt(se1**2 + se2**2)
        # calculate standard errors
        #se1, se2 = sem(x), sem(y)
        # standard error on the difference between the samples
        sed = sqrt(se1**2.0 + se2**2.0)
        # calculate the t statistic
        t_stat = (mean1 - mean2) / sed
        # degrees of freedom
        df = len(x) + len(y) - 2
        # calculate the critical value
        #cv = t.ppf(1.0 - alpha, df)
        # calculate the p-value
        p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
        # return everything
        return t_stat, df, p


if __name__ == '__main__':
    # alpha = sys.argv[1]
    # sample_size = sys.argv[2]
    # effect_size = sys.argv[3]
    # power = sys.argv[4]

    # b = 1 - (float(alpha)/2)
    # confidence_level = stats.norm.ppf(b)

    x = pd.Series(np.random.randint(0, 2, size=30))
    ab = ABTest(x)
    print(ab.z_statistic_for_two_proportions())

    # x = pd.Series(np.random.randint(10, 20, size=30))
    # y = pd.Series(np.random.randint(15, 30, size=30))
    # print(x.mean(), y.mean())

    # ab = ABTest(x, y)
    # print(ab.t_statistic_for_two_averages())
    # print(stats.ttest_ind(x, y, equal_var=False))
    # print(stats.ttest_ind(x, y, equal_var=True))
