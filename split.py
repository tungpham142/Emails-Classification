import pandas as pd

data = pd.read_csv('emails.csv')

test = data.iloc[1096:2243]

index = []
for i in range(1095):
    index.append(i)
for j in range(2244, 5630):
    index.append(j)

train = data.iloc[index]

#test.to_csv("test.csv", encoding='utf-8', index=False)
train.to_csv("train.csv", encoding='utf-8', index=False)