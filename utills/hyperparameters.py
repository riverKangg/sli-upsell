def get_xgb_hyperparameters():
    xgb_params = {
        'max_depth': (3, 10),
        'learning_rate': (0.01, 0.2),
        'n_estimators': (10, 300),
        'min_child_weight': (0, 10),
        'subsample': (0.5, 1),
        'colsample_bytree': (0.5, 1),
        'reg_alpha': (0, 10),
        'reg_lambda': (0, 10),
        'gamma': (0, 10)
    }
    return xgb_params

def get_lgb_hyperparameters():
    lgb_params = {
        'colsample_bytree': (0.5, 1),
        'learning_rate': (0.01, 0.2),
        'max_depth': (3, 20),
        'min_child_samples': (20, 1000),
        'n_estimators': (10, 500),
        'num_leaves': (5, 500),
        'subsample': (0.5, 1),
        'reg_alpha': (0, 10),
        'reg_lambda': (0, 10)
    }
    return lgb_params
def convert_params_to_int(params_dict):
    int_params = ['max_depth', 'n_estimators']
    for param_name in int_params:
        if param_name in params_dict:
            params_dict[param_name] = int(params_dict[param_name])
    return params_dict
