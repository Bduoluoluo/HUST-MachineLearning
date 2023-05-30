import numpy as np
import pandas as pd

def processTrain (data):
    '''
    特征 Age 分类：
    0: 小于12
    1: 大于等于12, 小于30
    2: 大于等于30, 小于60
    3: 大于等于60, 小于75
    4: 大于等于75
    5: 缺失
    '''
    data["Age"] = data["Age"].map (
        lambda x:
        0 if x < 12
        else 1 if x < 30
        else 2 if x < 60
        else 3 if x < 75
        else 4 if x >= 75
        else 5
    )

    '''
    特征 SibSp 分类：
    0: 小于1
    1: 大于等于1, 小于3
    2: 大于等于3
    '''
    data["SibSp"] = data["SibSp"].map (
        lambda x:
        0 if x < 1
        else 1 if x < 3
        else 2
    )

    '''
    特征 Parch 分类：
    0: 小于1
    1: 大于等于1, 小于4
    2: 大于等于4
    '''
    data["Parch"] = data["Parch"].map (
        lambda x:
        0 if x < 1
        else 1 if x < 4
        else 2
    )

    '''
    特征 Fare 分类：
    0: 对数化后小于1.5
    1: 对数化后大于等于1.5, 小于2.5
    2: 对数化后大于等于2.5
    '''
    data["Fare"] = data["Fare"].map (
        lambda x:
        0 if np.log(x + 1) < 1.5
        else 1 if np.log(x + 1) < 2.5
        else 2 
    )

    '''
    特征 Cabin 分类：
    0: 未缺失
    1: 缺失
    '''
    data["Cabin"] = data["Cabin"].map (
        lambda x:
        0 if type(x) == str
        else 1
    )

    # 特征 Embarked 的缺失数据用众数 S 填充
    data["Embarked"].fillna('S', inplace = True)

    '''
    特征 Embarked 分类：
    0: S
    1: C
    2: Q
    '''
    data["Embarked"].replace('S', 0, inplace = True)
    data["Embarked"].replace('C', 1, inplace = True)
    data["Embarked"].replace('Q', 2, inplace = True)

    y = data["Survived"]
    X = data.drop(['Survived', 'PassengerId', 'Name', 'Ticket'], axis = 1)
    X = pd.get_dummies(X)
    return X, y

def processTest (data):
    # 特征 Age 分类
    data["Age"] = data["Age"].map (
        lambda x:
        0 if x < 12
        else 1 if x < 30
        else 2 if x < 60
        else 3 if x < 75
        else 4 if x >= 75
        else 5
    )

    # 特征 SibSp 分类
    data["SibSp"] = data["SibSp"].map (
        lambda x:
        0 if x < 1
        else 1 if x < 3
        else 2
    )

    # 特征 Parch 分类
    data["Parch"] = data["Parch"].map (
        lambda x:
        0 if x < 1
        else 1 if x < 4
        else 2
    )

    # 特征 Fare 的缺失数据用均值填充
    data["Fare"].fillna(data["Fare"].mean(), inplace = True)

    # 特征 Fare 分类
    data["Fare"] = data["Fare"].map (
        lambda x:
        0 if np.log(x + 1) < 1.5
        else 1 if np.log(x + 1) < 2.5
        else 2 
    )

    # 特征 Cabin 分类
    data["Cabin"] = data["Cabin"].map (
        lambda x:
        0 if type(x) == str
        else 1
    )

    # 特征 Embarked 分类
    data["Embarked"].replace('S', 0, inplace = True)
    data["Embarked"].replace('C', 1, inplace = True)
    data["Embarked"].replace('Q', 2, inplace = True)

    id = data["PassengerId"]
    X = data.drop(['PassengerId', 'Name', 'Ticket'], axis = 1)
    X = pd.get_dummies(X)
    return id, X


