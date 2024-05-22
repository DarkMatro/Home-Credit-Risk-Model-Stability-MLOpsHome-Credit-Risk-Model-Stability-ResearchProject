import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, log_loss


def metrics_estimation(model, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series,
                       weeks_train: pd.Series, weeks_test: pd.Series, name: str) -> pd.DataFrame:
    """Generating tables with metrics for classification.

    Parameters
    ----------
    model: sklearn clf estimator
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    weeks_train: pd.Series
    weeks_test: pd.Series
    name: str
        id of metrics
    Returns
    -------
    df: pd.DataFrame
    """
    y_pred_train = model.predict(X_train)
    y_score_train = model.predict_proba(X_train)
    y_pred_test = model.predict(X_test)
    y_score_test = model.predict_proba(X_test)

    base_train = pd.DataFrame({'WEEK_NUM': weeks_train, 'target': y_train, 'score': y_score_train[:, 1]})
    base_test = pd.DataFrame({'WEEK_NUM': weeks_test, 'target': y_test, 'score': y_score_test[:, 1]})
    df_train = get_metrics(y_train, y_pred_train, y_score_train, name + '_train', base_train)
    df_test = get_metrics(y_test, y_pred_test, y_score_test, name + '_test', base_test)
    df = pd.concat([df_train, df_test])
    df.set_index('model', inplace=True)
    auc_train = df.loc[name + '_train']['ROC_AUC']
    auc_test = df.loc[name + '_test']['ROC_AUC']
    df['overfitting, %'] = (abs(auc_train - auc_test) / auc_test * 100)
    return df


def get_metrics(y_true: np.ndarray | pd.Series, y_pred: np.ndarray, y_score: np.ndarray, name: str,
                base: pd.DataFrame) -> pd.DataFrame:
    """
    Generating tables with metrics for classification.

    Parameters
    ----------
    y_true: 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_pred: 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.

    y_score: array-like of shape (n_samples,) or (n_samples, n_classes)
        Target scores.

    name: str
        id of metrics

    base: pd.DataFrame
        with 'WEEK_NUM', 'target' and 'score' columns

    Returns
    -------
    df: pd.DataFrame
    """
    df_metrics = pd.DataFrame()
    df_metrics['model'] = [name]
    df_metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    df_metrics['ROC_AUC'] = roc_auc_score(y_true, y_score if len(y_score.shape) == 1 else y_score[:, 1])
    df_metrics['Precision'] = precision_score(y_true, y_pred)
    df_metrics['Recall'] = recall_score(y_true, y_pred)
    df_metrics['f1'] = f1_score(y_true, y_pred)
    df_metrics['Logloss'] = log_loss(y_true, y_score)
    df_metrics['gini_stability'] = gini_stability(base)
    return df_metrics


def gini_stability(base: pd.DataFrame, w_fallingrate=88.0, w_resstd=-0.5) -> float:
    """
    Target metric of the competition.

    Parameters
    ----------
    base: pd.DataFrame
        with target, score and WEEK_NUM

    w_fallingrate: float
        Default=88.0

    w_resstd: float
        Default=-0.5

    Returns
    -------
    out: float
    """
    gini_in_time = base.loc[:, ["WEEK_NUM", "target", "score"]].sort_values("WEEK_NUM")
    gini_in_time = gini_in_time.groupby("WEEK_NUM")[["target", "score"]]
    gini_in_time = gini_in_time.apply(lambda df: 2 * roc_auc_score(df["target"], df["score"]) - 1).tolist()
    x = np.arange(len(gini_in_time))
    y = gini_in_time
    a, b = np.polyfit(x, y, 1)
    y_hat = a * x + b
    residuals = y - y_hat
    res_std = np.std(residuals)
    avg_gini = np.mean(gini_in_time)
    return avg_gini + w_fallingrate * min(0, a) + w_resstd * res_std
