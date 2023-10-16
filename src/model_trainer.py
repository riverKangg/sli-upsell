import os
import pickle
import warnings
import pandas as pd
from datetime import datetime
from unidecode import unidecode
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from model_optimizer import ModelOptimizer

warnings.filterwarnings(action='ignore')

class ModelTrainer:
    def __init__(self, trainset, testset, model_name='', col_lst=[]):
        self.trainset = trainset
        self.testset = testset
        self.model_name = model_name
        self.col_lst = list(set(col_lst))

        if model_name == '':
            model_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        print(f'Model Version\t {model_name}\n')

        result_path = './result/model'
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        self.path_bparams = f'{result_path}/{model_name}_bparams.xlsx'
        self.path_model = f"{result_path}/{model_name}.pkl"
        self.path_importance = f'{result_path}/{model_name}_importance.xlsx'

        self.col_version = 'none'
        check_cols = list(filter(lambda x: x not in trainset.columns, col_lst))
        if col_lst and not check_cols:
            self.col_version = 'before_dummy'
        elif col_lst and check_cols:
            self.col_version = 'after_dummy'

        if model_name not in ['xgb', 'lgb']:
            print('Model not recognized.')

        if list(trainset.columns) != list(testset.columns):
            print('Dataset mismatch.')

    def make_input(self):
        if self.col_version == 'before_dummy':
            up_dt = self.trainset[self.col_lst]
            up_test = self.testset[self.col_lst]
        else:
            up_dt = self.trainset
            up_test = self.testset

        drop_cols = ['마감년월', '계약자고객ID', '계약자주민등록번호암호화']
        up_dt = up_dt.drop(columns=drop_cols)
        up_test = up_test.drop(columns=drop_cols)

        null_cols = up_dt.columns[up_dt.isnull().any()].tolist()
        if null_cols:
            print(up_dt[null_cols].isnull().sum())
            up_dt = up_dt.fillna(0)
            up_test = up_test.fillna(0)

        up_dt2 = pd.get_dummies(up_dt)
        up_test2 = pd.get_dummies(up_test)

        for col in up_dt2.columns:
            if any(ord(c) > 127 for c in col):
                new_col = unidecode(col)
                up_dt2 = up_dt2.rename(columns={col: new_col})
                up_test2 = up_test2.rename(columns={col: new_col})

        if self.col_version == 'after_dummy':
            self.col_lst += ['PERF']
            up_dt2 = up_dt2[self.col_lst]
            up_test2 = up_test2[self.col_lst]

        x_dev = up_dt2.drop(columns=['PERF'])
        y_dev = up_dt2['PERF']
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(x_dev, y_dev, test_size=0.3, random_state=42)

        self.X_test = up_test2.drop(columns=['PERF'])
        self.y_test = up_test2['PERF']

    def train(self):
        self.make_input()
        if self.model_name == 'xgb':
            optimizer = ModelOptimizer(self.X_train, self.X_val, self.Y_train, self.Y_val, model_type='xgb')
        elif self.model_name == 'lgb':
            optimizer = ModelOptimizer(self.X_train, self.X_val, self.Y_train, self.Y_val, model_type='lgb')

        best_params = optimizer.optimize()

        if self.model_name == 'xgb':
            model = XGBClassifier(random_state=50, **best_params)
        elif self.model_name == 'lgb':
            model = LGBMClassifier(random_state=50, **best_params)

        model.fit(self.X_train, self.Y_train, eval_metric='auc', eval_set=[(self.X_val, self.Y_val)], early_stopping_rounds=100, verbose=0)

        importance_df = pd.DataFrame()
        importance_df['VAR'] = self.X_train.columns
        if self.model_name == 'xgb':
            importance_df['importance'] = model.feature_importances_
        elif self.model_name == 'lgb':
            importance_df['split'] = model.feature_importances_
            importance_df['gain'] = model.booster_.feature_importance(importance_type='gain')

        writer1 = pd.ExcelWriter(self.path_bparams, engine='xlsxwriter')
        pd.DataFrame.from_dict(best_params, orient='index').to_excel(writer1, index=False, encoding='utf-8-sig')
        writer1.save()

        writer2 = pd.ExcelWriter(self.path_importance, engine='xlsxwriter')
        importance_df.to_excel(writer2, index=False, encoding='utf-8-sig')
        writer2.save()

        with open(self.path_model, "wb") as f:
            pickle.dump(model, f)

        roc_score = roc_auc_score(self.y_test, model.predict_proba(self.X_test)[:, 1], average='macro')
        print(f'Test ROC \t {roc_score:.4f}')

# Example usage
trainset = pd.DataFrame()  # Replace with your actual trainset
testset = pd.DataFrame()   # Replace with your actual testset
model_trainer = ModelTrainer(trainset, testset, model_name='xgb', col_lst=['col1', 'col2', 'col3'])
model_trainer.train()
