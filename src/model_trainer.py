import os
import pickle
import warnings
import pandas as pd
from datetime import datetime
from unidecode import unidecode
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from utils.hyperparameters import get_xgb_hyperparameters, get_lgb_hyperparameters
from src.model_optimizer import ModelOptimizer

warnings.filterwarnings(action='ignore')

class ModelTrainer:
    def __init__(self, trainset, testset, model_type='xgb', col_lst=[]):
        self.trainset = trainset
        self.testset = testset
        if model_type not in ['xgb', 'lgb']:
            raise ValueError("Invalid model_type. Please use 'xgb' or 'lgb'.")
        self.model_type = model_type
        self.col_lst = list(set(col_lst))

        dt = datetime.now().strftime('%y%m%d_%H%M')
        save_name = f'{model_type}_{dt}'
        print(f'Model Version\t {save_name}\n')

        result_path = './result/model'
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        self.path_bparams = f'{result_path}/{save_name}_bparams.csv'
        self.path_model = f"{result_path}/{save_name}.pkl"
        self.path_importance = f'{result_path}/{save_name}_importance.csv'
        self.path_results = f'{result_path}/results.csv'
        self.model_name = save_name

        self.col_version = 'none'
        check_cols = list(filter(lambda x: x not in trainset.columns, col_lst))
        if col_lst and not check_cols:
            self.col_version = 'before_dummy'
        elif col_lst and check_cols:
            self.col_version = 'after_dummy'

        if model_type not in ['xgb', 'lgb']:
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

    def train(self, pbounds=None):
        self.make_input()
        if self.model_type == 'xgb':
            if pbounds==None:
                pbounds = get_xgb_hyperparameters()
            optimizer = ModelOptimizer(self.X_train, self.X_val, self.Y_train, self.Y_val,
                                       model_type='xgb', pbounds=pbounds)
        elif self.model_type == 'lgb':
            if pbounds==None:
                pbounds = get_lgb_hyperparameters()
            optimizer = ModelOptimizer(self.X_train, self.X_val, self.Y_train, self.Y_val,
                                       model_type='lgb', pbounds=pbounds)

        best_score, best_params = optimizer.optimize()

        if self.model_type == 'xgb':
            model = XGBClassifier(random_state=50, **best_params)
            model.fit(self.X_train, self.Y_train, eval_metric='auc', eval_set=[(self.X_val, self.Y_val)],
                      early_stopping_rounds=10, verbose=0)
        elif self.model_type == 'lgb':
            model = LGBMClassifier(random_state=50, **best_params)
            model.fit(self.X_train, self.Y_train, eval_metric='auc', eval_set=[(self.X_val, self.Y_val)],
                      callbacks=[early_stopping(10, verbose=False)])


        importance_df = pd.DataFrame()
        importance_df['VAR'] = self.X_train.columns
        if self.model_type == 'xgb':
            importance_df['importance'] = model.feature_importances_
        elif self.model_type == 'lgb':
            importance_df['split'] = model.feature_importances_
            importance_df['gain'] = model.booster_.feature_importance(importance_type='gain')

        best_params_df = pd.DataFrame.from_dict(best_params, orient='index')
        best_params_df.to_csv(self.path_bparams)
        importance_df.to_csv(self.path_importance, index=False)

        with open(self.path_model, "wb") as f:
            pickle.dump(model, f)

        roc_score = roc_auc_score(self.y_test, model.predict_proba(self.X_test)[:, 1], average='macro')
        print(f'Test ROC \t {roc_score:.4f}')

        results_df = pd.DataFrame({
            'Model': [self.model_name],
            'Validation ROC': [best_score],
            'Test ROC': [roc_score]
        })

        if os.path.exists(self.path_results):
            results_df.to_csv(self.path_results, mode='a', header=False, index=False)
        else:
            results_df.to_csv(self.path_results, index=False)

if __name__ == "__main__":
    trainset = pd.read_csv('../data/sample_data_202211.csv')
    testset = pd.read_csv('../data/sample_data_202304.csv')
    model_trainer = ModelTrainer(trainset, testset, model_type='xgb')
    model_trainer.train()