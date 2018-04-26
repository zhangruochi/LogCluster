import pandas as pd
import numpy as np


def plus(df, n):
    df['c'] = (df['a'] + df['b'])
    df['d'] = (df['a'] + df['b']) * n
    #return df


list1 = [[1, 3], [7, 8], [4, np.nan]]
df1 = pd.DataFrame(list1, columns=['a', 'b'])

df2 = pd.DataFrame(list1, columns=['a', 'b'])
print(df1[["a"]]/df2[["a"]])
