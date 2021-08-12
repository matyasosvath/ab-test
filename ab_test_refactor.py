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
- confidence interval

"""

ci_level = 95
power = 0.8
alpha = 0.05

alpha = 0.05
b = 1 - (float(alpha)/2)
confidence_level = ss.norm.ppf(b)
power = 0.8


class ABTest(object):
    """
    A simple AB Testing class.
    
    """


    def ci_props(self):
        pass

    def ci_avgs(self):
        pass


    def t_test(self, col1, col2): # calss ci_avgs
        print('T-test for two averages')
        print(self.df[[col1,col2]])

        #t_test_stat, p_value, ci_lower, ci_upper = 0,0,0,0
        #return t_test_stat, p_value, (ci_lower, ci_upper)

    def z_test(self, col1, col2): # calss ci_props
        
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

        
        print(self.df[[col1,col2]])

        #z_test_stat, p_value, (ci_lower, ci_upper) = 0,0,0,0
        #return z_test_stat, p_value, (ci_lower, ci_upper)
        

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

if __name__ == '__main__':
  fire.Fire(ABTest)