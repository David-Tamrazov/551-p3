import numpy as np
import pandas as pd

df_x = pd.read_csv("train_x.csv",nrows=1000)

df_x.to_csv("train_x_1000.csv",index = False)

df_y = pd.read_csv("train_y.csv",nrows=1000)

df_y.to_csv("train_y_1000.csv",index = False)
