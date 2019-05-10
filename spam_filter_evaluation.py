from spam_filter import spam_filter
import pandas as pd

spam_filter = spam_filter()

test = pd.read_csv('test.csv')
feature = test['text']
label = test['spam']
correct = 0

for i in range(len(feature)):
    query = feature[i]
    predict = spam_filter.classify(query)
    if predict == label[i]:
        correct += 1

accuracy = correct/float(len(label))

print("There are " + str(correct) +" correct classification in total of " + str(len(label)) + " cases")
print( str(accuracy * 100) + "%")
