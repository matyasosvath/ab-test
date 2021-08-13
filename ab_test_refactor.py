import fire
import scipy.stats as ss
import pandas as pd
import numpy as np

"""
4 moving parts
- sample size
- effect size
- power
- p-value
"""

#TODO Tests
#TODO error handling
#TODO var annotations
#TODO assert

class ABTest(object):
    """
    A simple AB Testing class.
    
    """
    def __init__(self):
        self.alpha=0.05
        self.b = 1 - (float(alpha)/2)
        self.power = 0.8

    def t_test(self, col1, col2, ci=True):
        print('T-test for two averages')
        #print(self.df[[col1,col2]])

        # Means
        mean1, mean2 = self.df[col1].mean(), self.df[col2].mean()

        # Calculate Standard error
        std1, std2 = self.df[col1].std(), self.df[col2].std()

        se1 = std1 / np.sqrt(self.df[col1].shape[0]) #teszt
        se2 = std2 / np.sqrt(self.df[col2].shape[0])

        standard_error_for_difference_between_means = np.sqrt(se1**2 + se2**2)
        
        
        t_test_statistic = (mean1 - mean2)/ standard_error_for_difference_between_means

        degrees_of_freedom = self.df[[col1, col2]].shape[0] - 2
        
        p_value = (1 - ss.t.cdf(abs(t_test_statistic), degrees_of_freedom)) * 2 # two-tailed
        
        # CONFIDENCE INTERVAL
        if ci:
            t_cl = np.round(ss.t.ppf(self.b, df=degrees_of_freedom),3) # t value for confidence interval

            ci_lower = mean1 - mean2 - t_cl * standard_error_for_difference_between_means
            ci_upper = mean1 - mean2 + t_cl * standard_error_for_difference_between_means

            return t_test_statistic, p_value, (ci_lower, ci_upper)

        else:
            return t_test_statistic, p_value

    def z_test(self, col1, col2), ci=True:
        
        print('Z-test for two proportions')

        prop_a, n_a = self.df[col1].value_counts(normalize=True)[1], len(self.df[col1])
        prop_b, n_b = self.df[col2].value_counts(normalize=True)[1], len(self.df[col2])
        prop_a, prop_b, n_a, n_b = float(prop_a), float(prop_b), float(n_a), float(n_b)
        
        print(prop_a, prop_b, n_a, n_b)
        
        p_comb = prop_a + prop_b
        print(p_comb)
        print(type(p_comb))

        z_score = (prop_b - prop_a) / np.sqrt(p_comb*(1-p_comb)*((1/n_a)+(1/n_b)))
        pvalue = ss.norm.pdf(abs(z_score)) * 2 # two-tailed
        return (z_score, pvalue) # abs(z_score) ???    

    def run(self, method, data, col1, col2):

        self.df = pd.read_csv(data, delimiter=',')
        print(self.df.head())
        
        if method=='average':
            return self.t_test(col1, col2)

        elif method=='props':
            return self.z_test(col1, col2)

        else:
            raise ValueError("Should not come here.")


# data = {'websiteA': np.random.randint(0,2, size=100),
#         'websiteB': np.random.randint(0,2, size=100),
#         }
#     #a = Assumptions()

# df = pd.DataFrame(data)
# df.to_csv('ab_test.csv', index=False)


# data = {'websiteA': np.random.randint(0,20, size=100),
#         'websiteB': np.random.randint(5,20, size=100),
#         }

# df = pd.DataFrame(data)
# df.to_csv('ab_test_avg.csv', index=False)



if __name__ == '__main__':
  fire.Fire(ABTest)