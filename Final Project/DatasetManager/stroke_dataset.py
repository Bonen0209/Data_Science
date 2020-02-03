import os
import warnings
import pandas as pd
import math
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, FunctionTransformer

DEBUG = True
IGNORE = True
if IGNORE:
    warnings.filterwarnings("ignore", category=FutureWarning)
from imblearn.over_sampling import RandomOverSampler, SMOTE

dir_path = os.path.abspath(os.path.dirname(__file__))
train_path = os.path.join(dir_path, './Dataset/train.csv')
test_path = os.path.join(dir_path, './Dataset/test.csv')

class StrokeDataset(Dataset):
    def __init__(self, train_path=train_path, test_path=test_path, transform=True, resample_method='ROSE', smoke='smoke'):
        super(StrokeDataset, self).__init__()

        self.train_path = train_path
        self.test_path = test_path
        self.transform = transform
        self.resample_method = resample_method
        self.smoke_dict = {'smoke': 0, 'no smoke': 1, 'fill nan in smokes': 2}
        self.smoke = self.smoke_dict[smoke]

        self.label_encoder = LabelEncoder()
        self.label_dict = {'hypertension': {'No': 0, 'Yes': 1}, 
                           'heart_disease': {'No': 0, 'Yes': 1},
                           'avg_glucose_level': {(0, 140): 0, (140, math.inf): 1},
                           'bmi': {(0, 18.5): 0,
                                   (18.5, 24): 1,
                                   (24, 27): 2,
                                   (27, 30): 3,
                                   (30, 35): 4,
                                   (35, math.inf): 5
                            },
                           'age': {(0, 5): 0,
                                   (5, 10): 1,
                                   (10, 15): 2,
                                   (15, 20): 3,
                                   (20, 25): 4,
                                   (25, 30): 5,
                                   (30, 35): 6,
                                   (35, 40): 7,
                                   (40, 45): 8,
                                   (45, 50): 9,
                                   (50, 55): 10,
                                   (55, 60): 11,
                                   (60, 65): 12,
                                   (65, 70): 13,
                                   (70, 75): 14,
                                   (75, 80): 15,
                                   (80, 85): 16,
                                   (85, math.inf): 17,
                            }
                        }

        self.train_data, \
        self.test_data = self._getData(self.train_path, self.test_path, transform=self.transform)

        self.train_data_with_smoke, \
        self.train_data_without_smoke, \
        self.test_data_with_smoke, \
        self.test_data_without_smoke = self._split_smoke(self.train_data, self.test_data)

        if self.smoke == 0 or self.smoke == 2:
            self.columns = self.train_data.columns[1: -1]
        elif self.smoke == 1:
            self.columns = self.train_data.columns[1: -2]

        self.X, self.Y, self.X_tree, self.Y_tree = self._choose(self.smoke)

    def _getData(self, train_path, test_path, transform):
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        train_data, test_data = self._fillna(train_data, test_data)

        if transform:
            columns = ['gender', 'ever_married', 'work_type', 'Residence_type']
            for column in columns:
                train_data[column] = self.label_encoder.fit_transform(train_data[column])
                test_data[column] = self.label_encoder.transform(test_data[column])

                self.label_dict[column] = dict(zip(self.label_encoder.classes_,
                                                self.label_encoder.transform(self.label_encoder.classes_)))

        print(f'Data loaded...')
        print(f'Train data shape: {train_data.shape}')
        print(f'Test data shape: {test_data.shape}\n')

        return train_data, test_data

    def _fillna(self, train_data, test_data):
        # Fill in nan with mean select by gender and age
        for low_age, high_age in self.label_dict['age']:
            for gender in ['Male', 'Female', 'Other']:
                condition = (train_data.age >= low_age) & \
                            (train_data.age < high_age) & \
                            (train_data.gender == gender)
                train_data.loc[condition, 'bmi'] = train_data.loc[condition, 'bmi'].fillna(train_data.loc[condition, 'bmi'].mean())

                condition = (test_data.age >= low_age) & \
                            (test_data.age < high_age) & \
                            (test_data.gender == gender)
                test_data.loc[condition, 'bmi'] = test_data.loc[condition, 'bmi'].fillna(test_data.loc[condition, 'bmi'].mean())
        
        return train_data, test_data

    def _split_smoke(self, train_data, test_data):
        train_data_with_smoke = train_data[train_data['smoking_status'].notnull()].reset_index(drop=True).copy()
        train_data_with_smoke['smoking_status'] = self.label_encoder.fit_transform(train_data_with_smoke['smoking_status'])
        self.label_dict['smoking_status'] = dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))
        train_data_without_smoke = train_data[train_data['smoking_status'].isnull()].reset_index(drop=True).copy()
        train_data_without_smoke.drop(columns='smoking_status', axis=1, inplace=True)

        test_data_with_smoke = test_data[test_data['smoking_status'].notnull()].reset_index(drop=True).copy()
        test_data_with_smoke['smoking_status'] = self.label_encoder.transform(test_data_with_smoke['smoking_status'])
        test_data_without_smoke = test_data[test_data['smoking_status'].isnull()].reset_index(drop=True).copy()
        test_data_without_smoke.drop(columns='smoking_status', axis=1, inplace=True)

        return train_data_with_smoke, train_data_without_smoke, test_data_with_smoke, test_data_without_smoke

    def _resample(self, data):
        if self.resample_method == 'ROSE':
            ros = RandomOverSampler()
            X, Y = ros.fit_resample(data.loc[:, ~data.columns.isin(['stroke', 'id'])], data['stroke'])
        elif self.resample_method == 'SMOTE':
            sm = SMOTE()
            X, Y = sm.fit_resample(data.loc[:, ~data.columns.isin(['stroke', 'id'])], data['stroke'])
        
        return X, Y

    def _choose(self, smoke):
        if smoke == 0:
            train_data = self.train_data_with_smoke
            train_data_tree = self._transform_tree(train_data.copy())

            X, Y = self._resample(train_data)
            X_tree, Y_tree = self._resample(train_data_tree)
        elif smoke == 1:
            train_data = self.train_data_without_smoke
            train_data_tree = self._transform_tree(train_data.copy())

            X, Y = self._resample(train_data)
            X_tree, Y_tree = self._resample(train_data_tree)
        elif smoke == 2:
            train_data = self.train_data.copy()
            train_data['smoking_status'] = train_data['smoking_status'].fillna('smokes')
            train_data['smoking_status'] = self.label_encoder.transform(train_data['smoking_status'])
            train_data_tree = self._transform_tree(train_data.copy())

            X, Y = self._resample(train_data)
            X_tree, Y_tree = self._resample(train_data_tree)

        return X, Y, X_tree, Y_tree
    
    def _transform_tree(self, data):
        # Turn values into range for tree base method
        data['age'] = data['age'].apply(self._label_age)
        data['avg_glucose_level'] = data['avg_glucose_level'].apply(self._label_avg_glucose_level)
        data['bmi'] = data['bmi'].apply(self._label_bmi)

        return data

    def _label_age(self, age):
        for low_age, high_age in self.label_dict['age']:
            if low_age <= age and age < high_age:
                return self.label_dict['age'][(low_age, high_age)]
    
    def _label_avg_glucose_level(self, avg_glucose_level):
        for low_avg_glucose_level, high_avg_glucose_level in self.label_dict['avg_glucose_level']:
            if low_avg_glucose_level <= avg_glucose_level and avg_glucose_level < high_avg_glucose_level:
                return self.label_dict['avg_glucose_level'][(low_avg_glucose_level, high_avg_glucose_level)]

    def _label_bmi(self, bmi):
        for low_bmi, high_bmi in self.label_dict['bmi']:
            if low_bmi <= bmi and bmi < high_bmi:
                return self.label_dict['bmi'][(low_bmi, high_bmi)]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def split(self, split=(1.0, 0.0), tree=False):
        if tree:
            X_train, X_test, Y_train, Y_test = train_test_split(self.X_tree, self.Y_tree, train_size=split[0], test_size=split[1])
        else:
            X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, train_size=split[0], test_size=split[1])
        
        return X_train, X_test, Y_train, Y_test

    def getDataLoader(self, split=(1.0, 0.0, 0.0), batch_size=32, num_workers=0):
        train_size = int(split[0] * self.__len__())
        valid_size = int(split[1] * self.__len__())
        test_size = self.__len__() - train_size - valid_size

        train, valid, test = random_split(self, [train_size, valid_size, test_size])

        return  DataLoader(train, batch_size=batch_size, shuffle=True ,num_workers=num_workers), \
                DataLoader(valid, batch_size=batch_size, num_workers=num_workers), \
                DataLoader(test, batch_size=batch_size, num_workers=num_workers)
        
    def description(self):
        print('Train data description')
        print(self.train_data.describe())
        print('Train data null description')
        print(self.train_data.isnull().sum() / len(self.train_data) * 100)
        print()

        print('Test data description')
        print(self.test_data.describe())
        print('Test data null description')
        print(self.test_data.isnull().sum() / len(self.test_data) * 100)
        print()

def main():
    if DEBUG:
        dataset = StrokeDataset()
   
if __name__ == "__main__":
    main()