import os
import pickle
import warnings
import pandas as pd
from datetime import datetime
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import roc_auc_score

from src.data_preprocessor import DataPreprocessor
from utils.hyperparameters import get_xgb_hyperparameters, get_lgb_hyperparameters
from src.model_optimizer import ModelOptimizer

warnings.filterwarnings(action='ignore')

class ModelTrainer:
    def __init__(self, X_train, X_val, Y_train, Y_val, X_test, Y_test, model_type='xgb'):
        if model_type not in ['xgb', 'lgb']:
            raise ValueError("Invalid model_type. Please use 'xgb' or 'lgb'.")
        self.model_type = model_type

        self.X_train, self.X_val, self.Y_train, self.Y_val = X_train, X_val, Y_train, Y_val
        self.X_test, self.Y_test = X_test, Y_test

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

        if model_type not in ['xgb', 'lgb']:
            print('Model not recognized.')

    def train(self, pbounds=None):
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

        roc_score = roc_auc_score(self.Y_test, model.predict_proba(self.X_test)[:, 1], average='macro')
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
    dp = DataPreprocessor(trainset, testset)
    X_train, X_val, Y_train, Y_val, X_test, Y_test = dp.process_data()
    model_trainer = ModelTrainer(X_train, X_val, Y_train, Y_val, X_test, Y_test , model_type='xgb')
    model_trainer.train()