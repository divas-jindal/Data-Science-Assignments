import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

# print("hello!")
# x= np.linspace(0,20,100)
# plt.plot(x,np.sin(x))
# plt.show()

#reading data
dataframe = pd.read_csv('data.csv',names=["R&D Spend","Administration","Marketing Spend","State","Profit"])

dataframe= (dataframe-dataframe.mean())/dataframe.std()
dataframe