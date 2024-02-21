from sklearn.preprocessing import MinMaxScaler
data = [[1.9,2,3,4]]
print(data)
scaler = MinMaxScaler()
d=scaler.fit_transform(data,columns=)
print(d)

import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
 
data = sns.load_dataset('iris')
print('Original Dataset')
data.head()
 
scaler = MinMaxScaler()
 
df_scaled = scaler.fit_transform(df.to_numpy())
df_scaled = pd.DataFrame(df_scaled, columns=[
  'sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
 
print("Scaled Dataset Using MinMaxScaler")
df_scaled.head()