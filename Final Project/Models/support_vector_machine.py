from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV
from Models.model import Model

class SupportVectorMachine(Model):
    def __init__(self, dataset):
        super(SupportVectorMachine, self).__init__(dataset, False)

        self.model = CalibratedClassifierCV(base_estimator=svm.LinearSVC)

    def __repr__(self):
        return 'Support Vector Machine'
    
    def run(self):
        self.model.fit(self.X_train, self.Y_train)

        return self.model
