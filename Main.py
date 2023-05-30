import pandas as pd
from DataProcess import processTrain, processTest
from DecisionTree import DecisionTreeClassifier
from LogisticRegression import LogisticRegressionClassifier


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X_train, y_train = processTrain(train)
id, X_test = processTest(test)



'''
逻辑回归
'''
lr_clf = LogisticRegressionClassifier(max_iter = 200, learning_rate = 0.01)
lr_clf.fit(X_train, y_train)
predictions = lr_clf.predict(X_test)

output = pd.DataFrame({"PassengerId": id, "Survived": predictions})
output.to_csv("LR_submission.csv", index = False)
'''
逻辑回归
'''



'''
决策树
'''
data = pd.concat([X_train, y_train], axis = 1, ignore_index = True)
dt_clf = DecisionTreeClassifier(epsilon = 0.141)
dt_clf.fit(data)
predictions = dt_clf.predict(X_test)

output = pd.DataFrame({"PassengerId": id, "Survived": predictions})
output.to_csv("DT_submission.csv", index = False)
'''
决策树
'''



'''
决策森林
'''
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators = 1400, max_depth = 4)

rf_clf.fit(X_train, y_train)
predictions = rf_clf.predict(X_test)

output = pd.DataFrame({"PassengerId": id, "Survived": predictions})
output.to_csv("RF_submission.csv", index = False)
'''
决策森林
'''