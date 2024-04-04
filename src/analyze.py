import matplotlib.pyplot as plt 
import pandas as pd 

plt.clf()
df_train = pd.read_csv('trainingEnv.csv')
plt.plot(df_train['Returns'])
plt.show()

plt.clf()
df_test = pd.read_csv('testingEnv.csv')
plt.plot(df_test['Returns'])
plt.show()