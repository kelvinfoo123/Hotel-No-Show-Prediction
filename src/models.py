from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, accuracy_score, recall_score, roc_auc_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

import optuna

def define_default_models(): 
    model_list = []

    logistic_regression = LogisticRegression()
    model_list.append(logistic_regression)

    knn = KNeighborsClassifier(n_neighbors=4)
    model_list.append(knn)

    rf = RandomForestClassifier()
    model_list.append(rf)

    xgb_classifier = xgb.XGBClassifier()
    model_list.append(xgb_classifier)

    return model_list

def run_default_models(X_train, X_test, y_train, y_test): 
    results = defaultdict(dict)
    models = define_default_models()

    for model in models: 
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        model_name = type(model).__name__
        results[model_name]['Accuracy'] = accuracy_score(y_test, y_pred)
        results[model_name]['Recall'] = recall_score(y_test, y_pred)
        results[model_name]['ROC-AUC'] = roc_auc_score(y_test, y_pred)

    model_list = []
    for k,v in results.items(): 
        print(f"{k}")
        model_list.append(k)
        for k, v in v.items(): 
            print(f"{k} - {v}")
        print("\n")

    recall_list = sorted(model_list, key = lambda x: (results[x]['Recall']), reverse = True)
    print(f"In descending order of recall score: {recall_list}")


class RandomForestTuner:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.study = None

    def tune_random_forest(self, trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 10),
            'n_jobs': -1
        }
        
        rf = RandomForestClassifier(**params)
        rf.fit(self.X_train, self.y_train)

        y_pred = rf.predict(self.X_test)
        recall = recall_score(self.y_test, y_pred)
        return recall

    def run_optuna(self):
        self.study = optuna.create_study(direction='maximize', study_name='rf_study')
        self.study.optimize(lambda trial: self.tune_random_forest(trial), n_trials=100, n_jobs=-1, show_progress_bar=True)

        print(f"Best test recall: {self.study.best_value:.2%}")
        for key, value in self.study.best_params.items():
            print(f"{key}: {value}")

    def run_best_model(self):
        best_rf = RandomForestClassifier(**self.study.best_params, n_jobs=-1)
        best_rf.fit(self.X_train, self.y_train)
        y_pred = best_rf.predict(self.X_test)
        print(f"Recall of tuned random forest model: {recall_score(self.y_test, y_pred)}")

def run_model(X_train, X_test, y_train, y_test): 
    define_default_models()
    run_default_models(X_train, X_test, y_train, y_test)
    best_predictor = RandomForestTuner(X_train, X_test, y_train, y_test)
    best_predictor.run_optuna()
    best_predictor.run_best_model()




