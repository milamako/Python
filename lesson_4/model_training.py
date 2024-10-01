from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

def train_and_evaluate(X, y):
    categorical_columns = ['Month', 'VisitorType']
    numerical_columns = [
        'Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration', 
        'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 
        'SpecialDay', 'OperatingSystems', 'Browser', 'Region', 'TrafficType'
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), numerical_columns),
            ('cat', OneHotEncoder(drop='first'), categorical_columns)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    feature_selector = SelectKBest(score_func=f_classif, k=10)

    def evaluate_model(model, param_grid):
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('feature_selector', feature_selector),
            ('model', model)
        ])
        
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=3, n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        print(f"Лучшие параметры для {model.__class__.__name__}: {grid_search.best_params_}")
        print(f"\nКлассификационный отчёт для {model.__class__.__name__}:")
        print(classification_report(y_test, y_pred))
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy для {model.__class__.__name__}: {accuracy}\n")

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Not Purchased', 'Purchased'], yticklabels=['Not Purchased', 'Purchased'])
        plt.title(f'Confusion Matrix for {model.__class__.__name__}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()

        mask = best_model.named_steps['feature_selector'].get_support()
        selected_features = np.array(numerical_columns + list(best_model.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_columns)))[mask]
        print(f"Наиболее значимые признаки для {model.__class__.__name__}: {selected_features}\n")
        
        return y_pred, model.__class__.__name__, accuracy

    param_grids = {
    'Gradient Boosting Classifier': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'model__n_estimators': [100, 300],
            'model__learning_rate': [0.05, 0.1],
            'model__max_depth': [3, 5, 7, 9],
            'model__min_samples_split': [2, 5, 10]
        }
    },
    'Random Forest Classifier': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'model__n_estimators': [100, 300],
            'model__max_depth': [None, 20],
            'model__min_samples_split': [2, 5, 10],
            'model__class_weight': ['balanced', 'balanced_subsample']
        }
    },
    'Extra Trees Classifier': {
        'model': ExtraTreesClassifier(random_state=42),
        'params': {
            'model__n_estimators': [100, 300],
            'model__max_depth': [None, 20],
            'model__min_samples_split': [2, 5, 10],
            'model__class_weight': ['balanced', 'balanced_subsample']
        }
    },
    'CatBoost Classifier': {
        'model': CatBoostClassifier(verbose=0, random_state=42),
        'params': {
            'model__iterations': [100, 300],
            'model__learning_rate': [0.05, 0.1],
            'model__depth': [4, 6, 8, 10],
            'model__class_weights': [[1, 5], [1, 10]],
            'model__l2_leaf_reg': [1, 3, 5, 7]
        }
    },
    'SVM Classifier': {
        'model': SVC(random_state=42),
        'params': {
            'model__C': [0.1, 1],
            'model__kernel': ['linear', 'rbf'],
            'model__class_weight': ['balanced', None]
        }
    },
    'XGBoost Classifier': {
        'model': XGBClassifier(random_state=42),
        'params': {
            'model__n_estimators': [100, 300],
            'model__learning_rate': [0.05, 0.1],
            'model__max_depth': [3, 5, 7, 9],
            'model__scale_pos_weight': [1, 5, 10]
        }
    }
}



    results = {}
    accuracies = []

    for model_name, model_info in param_grids.items():
        y_pred, model_name, accuracy = evaluate_model(model_info['model'], model_info['params'])
        results[model_name] = y_pred
        accuracies.append((model_name, accuracy))
    
    accuracies.sort(key=lambda x: x[1], reverse=True)
    model_names, model_accuracies = zip(*accuracies)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(model_accuracies), y=list(model_names), palette='viridis')
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Accuracy')
    plt.ylabel('Model')
    plt.show()

    return results
