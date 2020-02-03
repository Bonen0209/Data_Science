import os
from sklearn.linear_model import LogisticRegression as LogR
from sklearn.linear_model import LinearRegression as LinR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from Models.model import Model

cur_dir = os.path.abspath(os.path.dirname(__file__))

class LogisticRegression(Model):
    def __init__(self, dataset):
        super(LogisticRegression, self).__init__(dataset, False)

        self.model = LogR(max_iter=1000)
        self.filepath = os.path.join(cur_dir, './models/')

    def __repr__(self):
        return 'Logistic Regression'
    
    def run(self):
        self.model.fit(self.X_train, self.Y_train)

        return self.model

class LinearRegression(Model):
    def __init__(self, dataset):
        super(LinearRegression, self).__init__(dataset, False)

        self.model = LinR()
        self.filepath = os.path.join(cur_dir, './models/')

    def __repr__(self):
        return 'Linear Regression'

    def run(self):
        self.model.fit(self.X_train, self.Y_train)

        return self.model

class LinearDiscriminantAnalysis(Model):
    def __init__(self, dataset):
        super(LinearDiscriminantAnalysis, self).__init__(dataset,)

        self.model = LDA()
        self.filepath = os.path.join(cur_dir, './models/')

    def __repr__(self):
        return 'Linear Discriminant Analysis'

    def run(self):
        self.model.fit(self.X_train, self.Y_train)

        return self.model