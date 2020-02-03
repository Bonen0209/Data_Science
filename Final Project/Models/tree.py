import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from Models.model import Model

cur_dir = os.path.abspath(os.path.dirname(__file__))

class DecisionTree(Model):
    def __init__(self, dataset):
        super(DecisionTree, self).__init__(dataset, True)

        self.model = DecisionTreeClassifier()
        self.filepath = os.path.join(cur_dir, './models/')

    def __repr__(self):
        return 'Decision Tree'

    def run(self):
        self.model.fit(self.X_train, self.Y_train)

        return self.model

class RandomForest(Model):
    def __init__(self, dataset):
        super(RandomForest, self).__init__(dataset, True)

        self.model = RandomForestClassifier()
        self.filepath = os.path.join(cur_dir, './models/')
    
    def __repr__(self):
        return 'Random Forest'
    
    def run(self):
        self.model.fit(self.X_train, self.Y_train)

        return self.model

class KNearestNeighbors(Model):
    def __init__(self, dataset):
        super(KNearestNeighbors, self).__init__(dataset, True)

        self.model = KNeighborsClassifier()
        self.filepath = os.path.join(cur_dir, './models/')
    
    def __repr__(self):
        return 'K Nearest Neighbors'
    
    def run(self):
        self.model.fit(self.X_train, self.Y_train)

        return self.model
