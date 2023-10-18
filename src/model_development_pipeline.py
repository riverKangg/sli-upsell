import glob
import pickle
import pandas as pd

from utils import *
from src.model_trainer import ModelTrainer
from src.data_validator import DataValidator
from src.model_evaluator import ModelEvaluator
from src.score_calculator import ScoreCalculator
from src.data_preprocessor import DataPreprocessor
from src.variable_iv_psi_calculator import VariableIvPsi, DataBinningSummary

# Step 1: Load and check the dataset
train_data = pd.read_csv(f'{data_path}/sample_data_202211.csv')
is_train_data_valid, train_data_message = DataValidator(train_data).validate_modeling_data()

test_data = pd.read_csv(f'{data_path}/sample_data_202304.csv')
is_test_data_valid, test_data_message = DataValidator(test_data).validate_modeling_data()

if is_train_data_valid and is_test_data_valid:
    print("Data is valid.")

# Step 2: Data Preprocessing
data_processor = DataPreprocessor(train_data, test_data)
X_train, X_val, Y_train, Y_val, X_test, Y_test = data_processor.process_data()

# Step 3: Filtering by IV and PSI
binning_summary = DataBinningSummary(train_data, test_data)

binning_summary.make_categorical_binning_summary()
binning_summary.make_numerical_binning_summary()
binning_summary.make_total_binning_summary()

variable_iv_psi = VariableIvPsi(train_data, test_data, binning_summary.binning_summary_all_time)
variable_iv_psi.calculate_variable_iv()
variable_iv_psi.calculate_variable_psi()
variable_iv_psi.save_iv_psi()

# Step 4: Train the base model
model_trainer = ModelTrainer(X_train, X_val, Y_train, Y_val, X_test, Y_test, model_type='xgb')
model_trainer.train()

# Step 5: Filtering by importance and retraining
# If needed, filtering multiple times.

def get_latest_trained_model_and_importance(path=model_path, model_type='xgb'):
    model_list = glob.glob(f'{path}/{model_type}_*.pkl')
    latest_model = max(model_list)
    model_name = latest_model.split('/')[-1].replace('.pkl', '')

    importance = pd.read_csv(f'{model_path}/{model_name}_importance.csv')
    return model_name, importance

model_name, importance = get_latest_trained_model_and_importance()
importance = importance[importance.importance > 0]

filtered_columns = list(importance.VAR)

data_processor = DataPreprocessor(train_data, test_data, columns_to_keep=filtered_columns)
X_train, X_val, Y_train, Y_val, X_test, Y_test = data_processor.process_data()

model_trainer = ModelTrainer(X_train, X_val, Y_train, Y_val, X_test, Y_test, model_type='xgb')
model_trainer.train()

# Step 6: Check model performance and decide the final model
# Refer to modeling results in MODEL_RESULTS.csv file.
final_model_name = 'xgb_231018_074317'
with open(f'{model_path}/{final_model_name}.pkl', 'rb') as model_file:
    final_model = pickle.load(model_file)
final_model_variables = pd.read_csv(f'{model_path}/{final_model_name}_importance.csv').VAR.tolist()

data_processor = DataPreprocessor(train_data, test_data, columns_to_keep=final_model_variables)
X_train, X_val, Y_train, Y_val, X_test, Y_test = data_processor.process_data()
X_val = X_val[final_model_variables]
X_test = X_test[final_model_variables]

model_evaluator = ModelEvaluator(final_model, X_val, Y_val, X_test, Y_test)
validation_roc, test_roc = model_evaluator.calculate_roc_auc()
print(f'Validation ROC AUC: {validation_roc:.4f} / Test ROC AUC: {test_roc:.4f}')

psi, val_summary, test_summary, combined_summary, y_pred_val = model_evaluator.calculate_psi()
print(f'Population Stability Index (PSI): {psi:.4f}')
print(test_summary)

# Step 7: Scoring
score_data = pd.read_csv(f'{data_path}/sample_score_data_202306.csv')
score_variables = pd.read_csv(f'{model_path}/{final_model_name}_importance.csv').VAR.tolist()
score_calculator = ScoreCalculator('202306', score_data, final_model, score_variables)
score_calculator.get_score_and_grade()
score_calculator.save_score_df()