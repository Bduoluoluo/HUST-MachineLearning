from math import log
import numpy as np

class Node:
    def __init__ (self, root = True, label = None, feature_name = None, feature = None):
        self.root = root
        self.label = label
        self.feature_name = feature_name
        self.feature = feature
        self.tree = {}
        self.result = {
            "label": self.label,
            "feature": self.feature,
            "tree": self.tree
        }
    
    def __repr__ (self):
        return "{}".format(self.result)
    
    def add_node (self, val, node):
        self.tree[val] = node
    
    def predict (self, features):
        if self.root is True:
            return self.label
        return self.tree[features[self.feature]].predict(features)

class DecisionTreeClassifier:
    def __init__ (self, epsilon = 0.1):
        self.epsilon = epsilon
        self._tree = {}

    # 经验熵
    @staticmethod
    def calc_ent (datasets):
        data_length = len(datasets)
        label_count = {}
        for i in range(data_length):
            label = datasets[i][-1]
            if label not in label_count:
                label_count[label] = 0
            label_count[label] += 1
        ent = -sum([(p / data_length) * log(p / data_length, 2) for p in label_count.values()])
        return ent
    
    # 经验条件熵
    def cond_ent (self, datasets, axis = 0):
        data_length = len(datasets)
        feature_sets = {}
        for i in range(data_length):
            feature = datasets[i][axis]
            if feature not in feature_sets:
                feature_sets[feature] = []
            feature_sets[feature].append(datasets[i])
        cond_ent = sum([(len(p) / data_length) * self.calc_ent(p) for p in feature_sets.values()])
        return cond_ent
    
    # 信息增益
    @staticmethod
    def info_gain (ent, cond_ent):
        return ent - cond_ent
    
    def info_gain_train (self, datasets):
        count = len(datasets[0]) - 1
        ent = self.calc_ent(datasets)
        best_feature = []
        for c in range(count):
            c_info_gain = self.info_gain(ent, self.cond_ent(datasets, axis = c))
            best_feature.append((c, c_info_gain))
        best_ = max(best_feature, key = lambda x: x[-1])
        return best_
    
    def train (self, train_data):
        # _为特征值，y_train为类别值，features为特征名
        _, y_train, features = train_data.iloc[:, : -1], train_data.iloc[:, -1], train_data.columns[: -1]

        # 如果该节点所有数据类别值都相同，则将该值作为节点的类别值
        if len(y_train.value_counts()) == 1:
            return Node(root = True, label = y_train.iloc[0])
        
        # 如果该节点所有数据的特征都相同，则将类别值的众数作为节点的类别值
        if len(features) == 0:
            return Node(root = True, label = y_train.value_counts().sort_values(ascending = False).index[0])
        
        # 计算最大信息增益
        max_feature, max_info_gain = self.info_gain_train(np.array(train_data))
        max_feature_name = features[max_feature]

        # 如果最大信息增益小于阈值，则直接将类别值的众数作为节点的类别值
        if (max_info_gain < self.epsilon):
            return Node(root = True, label = y_train.value_counts().sort_values(ascending = False).index[0])
        
        node_tree = Node(root = False, feature_name = max_feature_name, feature = max_feature)

        # 该最大信息增益的特征的所有取值
        feature_list = train_data[max_feature_name].value_counts().index

        for f in feature_list:
            # 同一取值的划分
            sub_train_df = train_data.loc[train_data[max_feature_name] == f].drop([max_feature_name], axis = 1)
            
            # 递归构建子集生成树
            sub_tree = self.train(sub_train_df)
            node_tree.add_node(f, sub_tree)
        
        return node_tree
    
    def fit (self, train_data):
        self._tree = self.train(train_data)
        return self._tree
    
    def predict (self, X_test):
        predictions = []
        X_test = X_test.values
        for x in X_test:
            result = self._tree.predict(x)
            predictions.append(result)
        return predictions