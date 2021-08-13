# AB-Test

Simple AB-testing for statistical significance with confidence interval.

Can be used for testing the significance between two proportions or averages.

Incorporates the following statistical concepts:

- Z-Statistics for testing the significance between two proportions
- Two-sample (Independent Samples) T-test

- Standard error
- Standard error of proportions
- Standard error of the difference between the means

- Confidence Interval
- Confidence Interval for proportion
- Confidence Intervals for the difference between two proportions
- Confidence Interval for the difference between means


If you want to use it for count data. First, convert it to proportions, or use Chi-square.

Works from CLI

Run:

Proportions: `python3 ab_test.py run --method=props --data=filename.csv --col1=websiteA --col2=websiteB`
Averages: `python3 ab_test.py run --method=avgs --data=filename.csv --col1=websiteA --col2=websiteB`


Any questions arise, please contact me: osvath.matyas@med.unideb.hu

