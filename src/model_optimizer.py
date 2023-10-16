import warnings
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from bayes_opt import BayesianOptimization, UtilityFunction

warnings.filterwarnings(action='ignore')

class ModelOptimizer:
    def __init__(self, tr_x, val_x, tr_y, val_y, pbounds, n_iter, model_type):
        self.tr_x = tr_x
        self.tr_y = tr_y
        self.val_x = val_x
        self.val_y = val_y
        self.pbounds = pbounds
        self.n_iter = n_iter
        self.model_type = model_type

    def optimize(self):
        def objective(**params):
            if self.model_type == 'xgb':
                clf = XGBClassifier(eval_metric='logloss', n_jobs=-1, random_state=50, **params)
            else:
                clf = LGBMClassifier(boosting_type='goss', eval_metric='logloss', n_jobs=-1, random_state=50, **params)

            clf.fit(self.tr_x, self.tr_y, eval_set=[(self.val_x, self.val_y)], early_stopping_rounds=100, verbose=0)
            y_pred = clf.predict_proba(self.val_x)[:, 1]
            auc_score = roc_auc_score(self.val_y, y_pred, average='macro')
            return auc_score

        optimizer = BayesianOptimization(f=objective, pbounds=self.pbounds, verbose=2, random_state=1)
        optimizer.maximize(init_points=5, n_iter=self.n_iter, acquisition_function=UtilityFunction(kind='ei', xi=0.00))

        best_params = optimizer.max['params']
        best_score = optimizer.max['target']
        print("\n".join(f"{k}\t{v:.2f}" for k, v in best_params.items()))
        print(f"\nValidation AUC \t {best_score:.4f}")
        return best_params