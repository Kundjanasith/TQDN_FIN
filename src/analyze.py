import matplotlib.pyplot as plt 
import pandas as pd 

plt.subplot(121)
df_train = pd.read_csv('trainingEnv.csv')[:100]
plt.plot(df_train.index,df_train['Action'])

plt.subplot(122)
df_test = pd.read_csv('testingEnv.csv')[:100]
plt.plot(df_test.index,df_test['Action'])
plt.savefig('x.png',bbox_inches='tight')