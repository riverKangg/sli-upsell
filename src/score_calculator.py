import os
import pandas as pd
import numpy as np
import pickle
from utils.keys import data_keys
from utils.paths import *
from src.data_preprocessor import DataPreprocessor

class ScoreCalculator:
    def __init__(self, yyyymm, score_dataset, model, model_features, score_type='general'):
        """
        Initialize the ScoreCalculator.

        :param yyyymm: A string representing the year and month for scoring (e.g., '202306').
        :param score_dataset: The dataset used for scoring.
        :param model: The pre-trained model for making predictions.
        :param model_features: A list of model features used for scoring.
        :param score_type: The type of scoring ('general' or 'single').
        """
        if score_type not in ['general', 'single']:
            raise ValueError("Invalid score_type. Please use 'general' or 'single'.")
        self.yyyymm = yyyymm
        self.model = model
        self.data_keys = score_dataset[data_keys]

        data_preprocessor = DataPreprocessor(score_dataset)
        self.data_for_scoring = data_preprocessor.process_data()
        self.model_features = model_features
        self.score_type = score_type

    def compare_tables(self):
        """
        Compare and align the columns of data_for_scoring with the model features.
        """
        base_cols = set(self.model_features)
        compare_cols = set(self.data_for_scoring.columns.tolist())
        missing_cols = base_cols - compare_cols
        for col in missing_cols:
            self.data_for_scoring[col] = 0
        self.data_for_scoring = self.data_for_scoring[self.model_features]

    def get_score_df(self):
        """
        Calculate the probability of the model's predictions.
        """
        self.y_prob_df = pd.DataFrame(self.model.predict_proba(self.data_for_scoring)[:, 1], columns=['probability'])

    def score_transform(self, base_score=600, pdo=40, base_odds=1):
        """
        Transform the probabilities into scores using specified parameters.

        :param base_score: The base score for transformation.
        :param pdo: The points to double the odds (PDO).
        :param base_odds: The base odds.
        """
        self.y_prob_df['log_odds'] = np.log(self.y_prob_df['probability'] / (1 - self.y_prob_df['probability']))
        pdo_log_odds = pdo / np.log(2)
        self.y_prob_df['score'] = round(base_score + pdo_log_odds * self.y_prob_df['log_odds'], 0)
        self.y_pred_df = self.y_prob_df[:]
        del self.y_prob_df

    def get_score_and_grade(self):
        """
        Calculate scores, apply grade thresholds, and calculate statistics.
        """
        self.compare_tables()
        self.get_score_df()
        self.score_transform()
        self.y_pred_df['score'] = self.y_pred_df['score'].apply(lambda x: 1000 if x > 1000 else (1 if x < 1 else x))

        if self.score_type == 'general':
            score_conditions = [
                (self.y_pred_df['score'] >= 638),
                (self.y_pred_df['score'] >= 582) & (self.y_pred_df['score'] <= 637),
                (self.y_pred_df['score'] >= 538) & (self.y_pred_df['score'] <= 581),
                (self.y_pred_df['score'] >= 498) & (self.y_pred_df['score'] <= 537),
                (self.y_pred_df['score'] >= 457) & (self.y_pred_df['score'] <= 497),
                (self.y_pred_df['score'] >= 418) & (self.y_pred_df['score'] <= 456),
                (self.y_pred_df['score'] >= 322) & (self.y_pred_df['score'] <= 417),
                (self.y_pred_df['score'] >= 1) & (self.y_pred_df['score'] <= 321)
            ]
        elif self.score_type == 'single':
            score_conditions = [
                (self.y_pred_df['score'] >= 638),
                (self.y_pred_df['score'] >= 582) & (self.y_pred_df['score'] <= 637),
                (self.y_pred_df['score'] >= 538) & (self.y_pred_df['score'] <= 581),
                (self.y_pred_df['score'] >= 498) & (self.y_pred_df['score'] <= 537),
                (self.y_pred_df['score'] >= 457) & (self.y_pred_df['score'] <= 497),
                (self.y_pred_df['score'] >= 418) & (self.y_pred_df['score'] <= 456),
                (self.y_pred_df['score'] >= 322) & (self.y_pred_df['score'] <= 417),
                (self.y_pred_df['score'] >= 1) & (self.y_pred_df['score'] <= 321)
            ]

        self.y_pred_df['grade'] = np.select(score_conditions, [1, 2, 3, 4, 5, 6, 7, 8], default=0)

        score_df = self.y_pred_df.groupby('grade').size().reset_index(name='count')
        score_df['percentage'] = score_df['count'] / sum(score_df['count']) * 100
        score_df.columns = f'{self.score_type}_' + score_df.columns
        print(score_df)

        return self.y_pred_df

    def save_score_df(self):
        """
        Save the scored data to a CSV file.
        """
        if not os.path.exists(score_path):
            os.makedirs(score_path)
        score_data = self.data_keys.join(self.y_pred_df).set_index(data_keys)
        score_data.reset_index().to_csv(f"{score_path}/SCORE_{self.score_type}_{self.yyyymm}.csv", index=False)

if __name__ == "__main__":
    final_model_name = 'xgb_231017_150511'
    score_data = pd.read_csv(f'{data_path}/sample_score_data_202306.csv')
    with open(f'{model_path}/{final_model_name}.pkl', 'rb') as f:
        trained_model = pickle.load(f)
    variable_list = pd.read_csv(f'{model_path}/{final_model_name}_importance.csv').VAR.tolist()
    score_calculator = ScoreCalculator('202306', score_data, trained_model, variable_list)
    score_calculator.get_score_and_grade()
    score_calculator.save_score_df()
