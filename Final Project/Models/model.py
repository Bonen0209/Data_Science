import os
import torch
import joblib
import pandas as pd
from torch import nn
from abc import abstractmethod
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, \
                            accuracy_score, \
                            confusion_matrix, \
                            auc, \
                            roc_auc_score, \
                            precision_score, \
                            recall_score, \
                            roc_curve

cur_dir = os.path.abspath(os.path.dirname(__file__))
plot_path = os.path.join(cur_dir, '../Output/')

class Model(object):
    def __init__(self, dataset, tree):
        self.dataset = dataset
        self.model = None
        self.filepath = None

        self.X_train, \
        self.X_test, \
        self.Y_train, \
        self.Y_test = dataset.split(split=(0.8, 0.2), tree=tree)

    @abstractmethod
    def __repr__(self):
        pass

    def save(self):
        with open(f'{self.filepath}{self.__repr__}.pkl', 'wb') as f:  
            joblib.dump(self.model, f)

        print(f'Model {self.__repr__} saved...')

    def load(self):
        with open(f'{self.filepath}{self.__repr__}.pkl', 'rb') as f:  
            self.model = joblib.load(f)

        print(f'Model {self.__repr__} saved...')

    def plot_results(self, model_type, smoke, resample_method):
        Report = ''

        preds = self.model.predict(self.X_test)
        Report += f'Accuracy score: {accuracy_score(self.Y_test, preds)}\n\n'
        Report += f'Classification report\n{classification_report(self.Y_test, preds)}\n'
        Report += f'Confusion matrix\n{confusion_matrix(self.Y_test, preds)}\n\n'

        precision = precision_score(self.Y_test, preds)
        recall = recall_score(self.Y_test, preds)
        Report += f'precision: {precision}\n\nrecall: {recall}\n\n'

        y_pred_proba = self.model.predict_proba(self.X_test)[::, 1]
        fpr, tpr, _ = roc_curve(self.Y_test, y_pred_proba)
        auc = roc_auc_score(self.Y_test, y_pred_proba)
        plt.plot(fpr,tpr,label=f'{self.__repr__()} ({smoke},{resample_method}),\nauc={auc}')
        plt.legend(loc='lower right')
        plt.savefig(f'{plot_path}{self.__repr__()} ({smoke},{resample_method}).png')
        plt.close()

        if model_type in ['DST', 'RF']:
            impFeatures = pd.DataFrame(self.model.feature_importances_,
                                       index=self.dataset.columns,
                                       columns=['Importance']).sort_values(by='Importance',ascending=False)
            Report += impFeatures.__repr__()
        elif model_type in ['LogR']:
            impFeatures = pd.DataFrame(self.model.coef_[0],
                                       index=self.dataset.columns,
                                       columns=['Importance']).sort_values(by='Importance',ascending=False)
            Report += impFeatures.__repr__()

        Report += '\n'
        
        print(Report)

        with open(f'{plot_path}{self.__repr__()} ({smoke},{resample_method})', 'w') as f:
            f.write(Report)


class nn_Model(nn.Module):
    """
    Abstract model class
    """
    def __init__(self, dataset, smoke):
        super(nn_Model, self).__init__()
        self.filepath = None

    def forward(self):
        pass

    def save(self):
        """
        Saves the model
        :return: None
        """
        torch.save(self.state_dict(), self.filepath)
        print(f'Model {self.__repr__()} saved')

    def save_checkpoint(self, epoch_num):
        """
        Saves the model checkpoints
        :param epoch_num: int,
        :return: None
        """
        torch.save(self.state_dict(), self.filepath + '_' + str(epoch_num))
        print(f'Model checkpoint {self.__repr__()} saved for epoch')

    def load(self, cpu=False):
        """
        Loads the model
        :param cpu: bool, specifies if the model should be loaded on the CPU
        :return: None
        """
        if cpu:
            self.load_state_dict(torch.load(self.filepath, map_location=torch.device('cpu')))
        else:
            self.load_state_dict(torch.load(self.filepath, map_location=torch.device('cuda')))
        print(f'Model {self.__repr__()} loaded')
