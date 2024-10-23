from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import AdaBoostRegressor, VotingRegressor, StackingRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor, Lasso, LassoLars
import numpy as np

def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    """
    Возвращает:
        dict: Результаты моделирования.
    """
    models = {
        'DummyRegressor': {
            'model': DummyRegressor(),
            'params': {
                'strategy': ['mean', 'median', 'quantile', 'constant'],
                'quantile': [0.01, 0.1, 0.5, 1.0]
            }
        },
        'AdaBoostRegressor': {
            'model': AdaBoostRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.5, 1.0, 1.5],
                'loss': ['linear', 'square', 'exponential']
            }
        },
        'HuberRegressor': {
            'model': HuberRegressor(),
            'params': {
                'epsilon': [1.1, 1.35, 1.5, 1.75],
                'alpha': [1e-5, 1e-4, 1e-3, 1e-2],
                'max_iter': [100, 500, 1000],
                'tol': [1e-5, 1e-4, 1e-3],
                'fit_intercept': [True, False],
                'warm_start': [False, True]
            }
        },
        'Lasso': {
            'model': Lasso(random_state=42),
            'params': {
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
                'max_iter': [1000, 5000, 10000],
                'tol': [1e-4, 1e-3, 1e-2],
                'selection': ['cyclic', 'random'],
                'fit_intercept': [True, False],
                'copy_X': [True, False],
                'warm_start': [True, False],
                'positive': [True, False]
            }
        },
        'LassoLars': {
            'model': LassoLars(random_state=42),
            'params': {
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
                'max_iter': [500, 1000, 2000],
                'eps': [1e-16, 1e-10, 1e-8],
                'fit_intercept': [True, False],
                'verbose': [True, False],
                'positive': [False, True]
            }
        },
    }
    
    results = {}
    best_estimators = {}

    for name, config in models.items():
        print(f"\nОбучение модели: {name}")
        if config['params']:
            grid = GridSearchCV(
                estimator=config['model'],
                param_grid=config['params'],
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            best_estimators[name] = best_model
            print(f"Лучшие параметры: {grid.best_params_}")
        else:
            best_model = config['model']
            best_model.fit(X_train, y_train)
        
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)
        results[name] = {
            'model': best_model,
            'y_pred': y_pred,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2 Score: {r2:.4f}")

    voting_regressor = VotingRegressor(
        estimators=[
            ('lasso', best_estimators['Lasso']),
            ('huber', best_estimators['HuberRegressor']),
            ('adaboost', best_estimators['AdaBoostRegressor'])
        ]
    )
    voting_regressor.fit(X_train, y_train)
    y_pred = voting_regressor.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    results['VotingRegressor'] = {
        'model': voting_regressor,
        'y_pred': y_pred,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }

    print(f"\nVotingRegressor:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    stacking_regressor = StackingRegressor(
        estimators=[
            ('lasso', best_estimators['Lasso']),
            ('huber', best_estimators['HuberRegressor']),
            ('adaboost', best_estimators['AdaBoostRegressor'])
        ],
        final_estimator=RandomForestRegressor(n_estimators=100, random_state=42)
    )
    stacking_regressor.fit(X_train, y_train)
    y_pred = stacking_regressor.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    results['StackingRegressor'] = {
        'model': stacking_regressor,
        'y_pred': y_pred,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }

    print(f"\nStackingRegressor:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    return results
