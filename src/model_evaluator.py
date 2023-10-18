import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from utils import *
from src.data_preprocessor import DataPreprocessor


class ModelEvaluator:
    def __init__(self, model, val_x, val_y, te_x, te_y):
        """
        Initializes the ModelEvaluator.

        :param model: The machine learning model.
        :param val_x: Validation dataset features.
        :param val_y: Validation dataset target.
        :param te_x: Test dataset features.
        :param te_y: Test dataset target.
        """
        self.model = model
        self.val_x, self.val_y, self.te_x, self.te_y = val_x, val_y, te_x, te_y

    def calculate_roc_auc(self):
        """
        Calculates ROC AUC scores for the validation and test datasets.

        :return: Validation ROC AUC score, Test ROC AUC score.
        """
        validation_roc_auc = roc_auc_score(self.val_y, self.model.predict_proba(self.val_x)[:, 1], average='macro')
        test_roc_auc = roc_auc_score(self.te_y, self.model.predict_proba(self.te_x)[:, 1], average='macro')
        return validation_roc_auc, test_roc_auc

    def calculate_psi(self):
        """
        Calculates the Population Stability Index (PSI) and related statistics.

        :return: PSI value, Validation dataset summary, Test dataset summary, Combined summary, Validation dataset predictions.
        """
        y_pred_val = pd.DataFrame(self.model.predict_proba(self.val_x)[:, 1], columns=['probability'])
        y_pred_test = pd.DataFrame(self.model.predict_proba(self.te_x)[:, 1], columns=['probability'])

        # Define conditions for calculating PSI bins

        scale = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        conditions1 = [
            (y_pred_val >= 0.9),
            (y_pred_val >= 0.8) & (y_pred_val < 0.9),
            (y_pred_val >= 0.7) & (y_pred_val < 0.8),
            (y_pred_val >= 0.6) & (y_pred_val < 0.7),
            (y_pred_val >= 0.5) & (y_pred_val < 0.6),
            (y_pred_val >= 0.4) & (y_pred_val < 0.5),
            (y_pred_val >= 0.3) & (y_pred_val < 0.4),
            (y_pred_val >= 0.2) & (y_pred_val < 0.3),
            (y_pred_val >= 0.1) & (y_pred_val < 0.2),
            (y_pred_val >= 0) & (y_pred_val < 0.1)
        ]

        conditions2 = [
            (y_pred_test >= 0.9),
            (y_pred_test >= 0.8) & (y_pred_test < 0.9),
            (y_pred_test >= 0.7) & (y_pred_test < 0.8),
            (y_pred_test >= 0.6) & (y_pred_test < 0.7),
            (y_pred_test >= 0.5) & (y_pred_test < 0.6),
            (y_pred_test >= 0.4) & (y_pred_test < 0.5),
            (y_pred_test >= 0.3) & (y_pred_test < 0.4),
            (y_pred_test >= 0.2) & (y_pred_test < 0.3),
            (y_pred_test >= 0.1) & (y_pred_test < 0.2),
            (y_pred_test >= 0) & (y_pred_test < 0.1)
        ]

        y_pred_val['scale'] = np.select(conditions1, scale, default=0)
        y_pred_val = pd.merge(y_pred_val, self.val_y, left_index=True, right_index=True)
        y_pred_test['scale'] = np.select(conditions2, scale, default=0)
        y_pred_test = pd.merge(y_pred_test, self.te_y, left_index=True, right_index=True)

        # Calculate PSI and related statistics

        val_summary = y_pred_val.groupby('scale').size().reset_index(name='sample_count')
        val_target = y_pred_val.groupby('scale')['PERF'].sum().reset_index(name='target_count')
        val_summary = val_summary.merge(val_target, on='scale')
        val_summary['composition_ratio'] = val_summary['sample_count'] / val_summary['sample_count'].sum()
        val_summary['target_rate'] = val_summary['target_count'] / val_summary['sample_count']

        test_summary = y_pred_test.groupby('scale').size().reset_index(name='sample_count')
        test_target = y_pred_test.groupby('scale')['PERF'].sum().reset_index(name='target_count')
        test_summary = test_summary.merge(test_target, on='scale')
        test_summary['composition_ratio'] = test_summary['sample_count'] / test_summary['sample_count'].sum()
        test_summary['target_rate'] = test_summary['target_count'] / test_summary['sample_count']

        combined_summary = pd.merge(val_summary, test_summary, how='left', left_on='scale', right_on='scale')

        psi_values = (combined_summary['composition_ratio_x'] - combined_summary['composition_ratio_y']) * np.log(
            combined_summary['composition_ratio_x'] / combined_summary['composition_ratio_y'])
        psi = psi_values.sum()
        return psi, val_summary, test_summary, combined_summary, y_pred_val

if __name__ == "__main__":
    trainset = pd.read_csv(f'{data_path}/sample_data_202211.csv')
    testset = pd.read_csv(f'{data_path}/sample_data_202304.csv')

    final_model_name = 'xgb_231017_150511'
    with open(f'{model_path}/{final_model_name}.pkl', 'rb') as f:
        model = pickle.load(f)
    variable_list = pd.read_csv(f'{model_path}/{final_model_name}_importance.csv').VAR.tolist()

    dp = DataPreprocessor(trainset, testset, columns_to_keep=variable_list)
    X_train, X_val, Y_train, Y_val, X_test, Y_test = dp.process_data()
    X_val = X_val[variable_list]
    X_test = X_test[variable_list]
    print(X_val.shape, len(variable_list))

    me = ModelEvaluator(model, X_val, Y_val, X_test, Y_test)
    val_roc, test_roc = me.calculate_roc_auc()
    print(f'Val Roc: {val_roc:.4f} / Tset Roc: {test_roc:.4f}')
    psi, val_summary, test_summary, combined_summary, y_pred_val = me.calculate_psi()
    print(f'\nPSI: {psi:.4f}')
    print(test_summary)