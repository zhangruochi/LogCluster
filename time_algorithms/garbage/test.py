import pandas as pd

df2 = pd.DataFrame([[1,2],[3,4]],columns=['a', 'b'],index = ["c","d"])
print(df2)
print(df2.loc[["c","d"]])


df2["a"] = 2
print(df2)



