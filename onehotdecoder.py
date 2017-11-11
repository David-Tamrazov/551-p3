import pandas as pd

prediction = pd.read_csv("cnnresults.csv",index_col=0,delimiter=',',header=0)
print prediction

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81]

for index, rows in prediction.iterrows():
    rows['Label'] = classes[rows['Label']]

prediction.to_csv('cnnresults_p.csv',index=True)