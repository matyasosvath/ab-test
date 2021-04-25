#!/usr/local/bin/python

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


alpha = sys.argv[1]


if sys.argv:
	for i in sys.argv:
		print(i)
# sample_size = sys.argv[1]
# effect_size = sys.argv[2]
# power = sys.argv[3]


b = 1 - (float(alpha)/2)
confidence_level = stats.norm.ppf(b)


class ABTest:
 	def __init__(self, data, calculate='proportion'):
 		self.data = np.array(data)
 		self.n = len(data) # sample size


 	def prop():
 		csoportok = pd.value_counts(self.data)




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
 		sample_var =  sum(x-sample_mean for x in self.data)/self.n
 		print(sample_mean, sample_var)

 		se = sample_var/self.n # Standard Error
 		avg_minus = sample_mean - confidence_level*np.sqrt(se)
 		avg_plus = sample_mean + confidence_level*np.sqrt(se)

 		return (avg_minus, avg_plus)


 	def z_statistic_for_two_proportions(self):

 		X_a = len(p_a_converted)
 		X_b = len(p_b_converted)


 		se_pool = (1/n_a+1/n_b)

 		p = (X_a + X_b)/n_b + n_a
 		z_score = (p_a_converted - p_b_converted) / np.sqrt(p(1-p)(1/n_a+1/n_b))

 		pvalue = stats.norm.pdf(z_score) * 2
 		return (z_score, pvalue)


 	def t_statistic_for_two_averages(self):
 		pass




if __name__ == '__main__':
	x = list(range(15))
	y = list(range(10))

	ab = ABTest(x)
	print(ab.confidence_interval_average())

 	# if proportion:
 	# 	pass
 	# 	# proportions()


 	# if average:
 	# 	# avrage()


 	# def two_proportions(self, sample_a, sample_b):
 	# 	# calc p_a, p_b
 	# 	pass


 	# def two_averages(self, sample_a, sample_b):
 	# 	# calc averages
 	# 	# t-stat






# class TestSum(unittest.TestCase):

#     def test_calc_power(self):
#         self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")

#     def test_calc_confidence_interval(self):
#         self.assertEqual(sum((1, 2, 2)), 6, "Should be 6")

#     def test_calc_pvalue(self):
#     	pass
#     # ...

# if __name__ == '__main__':
#     unittest.main()

