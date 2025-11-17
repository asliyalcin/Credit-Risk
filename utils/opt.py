import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression


def evaluate_with_gmean(y_true, y_proba, print_results=True):
    """
    Finds the best threshold based on G-Mean and calculates performance metrics.
    
    Parameters
    ----------
    y_true : array-like
        True labels (0/1).
    y_proba : array-like
        Predicted probabilities for the positive class (model.predict_proba(... )[:,1]).
    print_results : bool
        If True, prints the results.

    Returns
    -------
    best_threshold : float
        Best threshold according to G-Mean.
    metrics : dict
        Contains tn, fp, fn, tp, sensitivity, specificity, and g_mean.
    y_pred : np.array
        Class predictions (0/1) generated using the best threshold.
    """
    # ROC curve and G-Mean calculation
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    best_threshold = thresholds[ix]

    # Predictions with the new threshold
    y_pred = (y_proba >= best_threshold).astype(int)

    # Performance metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    g_mean = np.sqrt(sensitivity * specificity)

    metrics = {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "g_mean": g_mean,
    }

    if print_results:
        print(f"Best threshold (based on G-Mean): {best_threshold:.3f}")
        print(
            f"Sensitivity: {sensitivity:.3f}, "
            f"Specificity: {specificity:.3f}, "
            f"G-Mean: {g_mean:.3f}"
        )

    return best_threshold, metrics, y_pred





def eval_func(y_true, y_pred):
    """
    Calculates TN, FP, FN, TP, Sensitivity, Specificity, and G-Mean using the confusion matrix.
    
    Parameters
    ----------
    y_true : array-like
        True labels (0/1).
    y_pred : array-like
        Model predictions (0/1).

    Returns
    -------
    metrics : dict
        TN, FP, FN, TP, sensitivity, specificity, g_mean.
    """

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Extracting metrics
    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    # Sensitivity (Recall / True Positive Rate)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    # Specificity (True Negative Rate)
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    # G-Mean
    g_mean = np.sqrt(sensitivity * specificity)

    metrics = {
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "TP": TP,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "g_mean": g_mean
    }

    return metrics





def objective_xgb(trial, scale_pos_weight, X_train_xbg, X_valid_xgb, y_train_xgb, y_valid_xgb, X):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', scale_pos_weight*0.5, scale_pos_weight*1.5),
        'eval_metric': 'logloss',
        'random_state': 42
    }

    # Feature selection
    mask = [trial.suggest_categorical(f"feature_{i}", [0, 1]) for i in range(X_train_xbg.shape[1])]
    selected_features = [col for col, m in zip(X.columns, mask) if m == 1]

    if len(selected_features) == 0:
        return 1e6

    X_train_sel = X_train_xbg[selected_features]
    X_valid_xgb_sel = X_valid_xgb[selected_features]

    model = XGBClassifier(**params)
    model.fit(X_train_sel, y_train_xgb,
              eval_set=[(X_valid_xgb_sel, y_valid_xgb)],
              verbose=False)

    y_pred = model.predict(X_valid_xgb_sel)
    
    tn, fp, fn, tp = confusion_matrix(y_valid_xgb, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    g_mean = np.sqrt(sensitivity * specificity)

    trial.set_user_attr("selected_features", selected_features)
    return g_mean



def objective_lgr(trial, X_train_lgr, y_train_lgr, X_valid_lgr, y_valid_lgr):
    # 1. Hyperparameter selections
    C = trial.suggest_float('C', 1e-3, 10.0, log=True)
    penalty_choice = trial.suggest_categorical('penalty', ['l2', 'none'])
    solver = trial.suggest_categorical('solver', ['lbfgs', 'saga'])

    # Convert penalty string to None for LogisticRegression
    penalty = None if penalty_choice == 'none' else penalty_choice

    # Model
    model = LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        max_iter=2000,
        class_weight='balanced',
        random_state=42
    )

    # Fit
    model.fit(X_train_lgr, y_train_lgr)

    # Probability prediction
    y_proba_valid = model.predict_proba(X_valid_lgr)[:, 1]

    # G-Mean calculation
    fpr, tpr, thresholds = roc_curve(y_valid_lgr, y_proba_valid)
    gmeans = np.sqrt(tpr * (1 - fpr))

    # Return best G-Mean
    return np.max(gmeans)



