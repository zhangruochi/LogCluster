
import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(10,6))

df.iloc[1:3,1] = np.nan
df.iloc[5,3] = np.nan
df.iloc[7:9,5] = np.nan
print(df)

print(df.isnull().any())