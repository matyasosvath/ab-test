#!/usr/bin/env python3

import pandas as pd
import numpy as np
import scipy.stats as ss
import fire
import logging


logger = logging.getLogger()
logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler() # messages show up in terminal
formatter = logging.Formatter('%(asctime)s %(levelname)s :: %(message)s') # format the message for the terminal output

stream_handler.setFormatter(formatter) # add formatter to the stream handler
stream_handler.setLevel(logging.INFO)

logger.addHandler(stream_handler)


class ABTest(object):
    """
    A simple AB Test for two proportions or averages.
    """
    def __init__(self):
        self.alpha=0.05
        self.__b = 1 - (float(self.alpha)/2)
        self.power = 0.8

        logging.info("AB Test class initialized!")

    def __t_test(self, col1, col2, ci=True):
        """
        Two-sample (Independent Samples) T-test (two-tailed)

        Input:

        col1: pandas.Series
        col2: pandas.Series

        Return

        t_test_statistic: T test statistic
        p_value: P-value for hypothesis test
        ci_lower: Confidence Interval Lower limit 
        ci_upper: Confidence Interval Upper limit
        """

        assert type(self.df[col1]) == pd.core.series.Series, "Col1 Should be pandas.Series"
        assert type(self.df[col2]) == pd.core.series.Series, "Col1 Should be pandas.Series"

        logging.info("Two-sample (Independent Samples) T-test (two-tailed) method running!")
        
        # Means
        mean1, mean2 = self.df[col1].mean(), self.df[col2].mean()

        # Calculate Standard error
        std1, std2 = self.df[col1].std(), self.df[col2].std()

        se1 = std1 / np.sqrt(self.df[col1].shape[0])
        se2 = std2 / np.sqrt(self.df[col2].shape[0])

        standard_error_for_difference_between_means = np.sqrt(se1**2 + se2**2)
        
        mean_diff = abs(mean1 - mean2)
        t_test_statistic = np.round((mean_diff / standard_error_for_difference_between_means),3)

        degrees_of_freedom = self.df[[col1, col2]].shape[0] - 2
        
        p_value = np.round((1 - ss.t.cdf(abs(t_test_statistic), degrees_of_freedom)) * 2, 3) # two-tailed
        
        # CONFIDENCE INTERVAL
        if ci:
            t_cl = ss.t.ppf(self.__b, df=degrees_of_freedom) # t value for confidence interval

            ci_lower = mean_diff - t_cl * standard_error_for_difference_between_means
            ci_upper = mean_diff + t_cl * standard_error_for_difference_between_means

            return t_test_statistic, p_value, np.round((ci_lower, ci_upper), 3)

        else:
            return t_test_statistic, p_value

    def __z_test(self, col1, col2, ci=True):
        """
        Z-test for two proportions

        Input:

        col1: pandas.Series
        col2: pandas.Series

        Return

        z_test_statistic: z test statistic
        p_value: P-value for hypothesis test
        ci_lower: Confidence Interval Lower limit 
        ci_upper: Confidence Interval Upper limit
        """

        assert type(self.df[col1]) == pd.core.series.Series, "Col1 Should be pandas.Series"
        assert type(self.df[col2]) == pd.core.series.Series, "Col1 Should be pandas.Series"

        logging.info("Z-test for two proportions method running!")

        prop_a, n_a = self.df[col1].value_counts(normalize=True)[1], len(self.df[col1])
        prop_b, n_b = self.df[col2].value_counts(normalize=True)[1], len(self.df[col2])
        prop_a, prop_b, n_a, n_b = float(prop_a), float(prop_b), float(n_a), float(n_b)
        
        # Standard error of two proportions
        se1 = np.sqrt((prop_a*(1-prop_a))/n_a)
        se2 = np.sqrt((prop_b*(1-prop_b))/n_b)

        standard_error_for_difference_between_proportions = np.sqrt(se1**2 + se2**2)
        
        prop_diff = abs(prop_b - prop_a)
        z_test_statistic = np.round((prop_diff / standard_error_for_difference_between_proportions),3)
        
        pvalue = np.round((ss.norm.pdf(abs(z_test_statistic)) * 2),3) # two-tailed
        
        # CONFIDENCE INTERVAL
        if ci:
            z_cl = ss.norm.ppf(self.__b)
            ci_lower = prop_diff - z_cl * standard_error_for_difference_between_proportions
            ci_upper = prop_diff + z_cl * standard_error_for_difference_between_proportions
            return z_test_statistic, pvalue, np.round((ci_lower, ci_upper), 3)
        
        else:
            return z_test_statistic, pvalue

    def run(self, method: str, data: pd.DataFrame, col1: str, col2: str) -> list:
        """
        Run:
        python3 ab_test.py run --method=props --data=ab_test_prop.csv --col1=websiteA --col2=websiteB
        python3 ab_test.py run --method=avgs --data=ab_test_avg.csv --col1=websiteA --col2=websiteB
        """

        try:
            self.df = data
        except (ValueError, TypeError):
            pass

        try:
            self.df = pd.read_csv(data, delimiter=',')    
        except (KeyError, ValueError):
            #print('Delimeter maybe wrong')
            pass

        if method=='avgs':
            return self.__t_test(col1, col2)

        elif method=='props':
            return self.__z_test(col1, col2)

        else:
            raise ValueError("Should not come here.")


# TESTS
import unittest

class TestABTest(unittest.TestCase):

    def setUp(self) -> None:
        np.random.seed(42)

        data = {'nominal1': np.random.randint(0,2, size=100),
            'nominal2': np.random.randint(0,2, size=100),
            'interval1': np.random.randint(0,20, size=100),
            'interval2': np.random.randint(0,20, size=100)
            }

        self.data = pd.DataFrame(data)
        self.abtest = ABTest()

    def test_t_test(self):
        t, p, ci = self.abtest.run('avgs', self.data, 'interval1', 'interval2')
    
        self.assertEqual(t, 0.422, "T test statistic error")
        self.assertEqual(p, 0.674, "Pvalue is not looking good")
        self.assertEqual(ci[0], -1.405, 'CI problem')

    def test_z_test(self):
        z, p, ci = self.abtest.run('props', self.data, 'nominal1', 'nominal2')

        self.assertEqual(z, 1.709, "T test statistic error")
        self.assertEqual(p, 0.185, "Pvalue is not looking good")
        self.assertEqual(ci[0], -0.018, 'CI problem')

if __name__ == '__main__':
  fire.Fire(ABTest)