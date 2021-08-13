import numpy as np
import pandas as pd
np.random.seed(42)

data = {'websiteA': np.random.randint(0,2, size=100),
        'websiteB': np.random.randint(0,2, size=100),
        }
df = pd.DataFrame(data)
df.to_csv('ab_test.csv', index=False)


data = {'websiteA': np.random.randint(0,20, size=100),
        'websiteB': np.random.randint(5,20, size=100),
        }
df = pd.DataFrame(data)
df.to_csv('ab_test_avg.csv', index=False)


data = {'nominal1': np.random.randint(0,2, size=100),
        'nominal2': np.random.randint(0,2, size=100),
        'interval1': np.random.randint(0,20, size=100),
        'interval2': np.random.randint(0,20, size=100)
        }

df = pd.DataFrame(data)

df.to_csv('scores.csv', index=False)