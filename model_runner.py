import uuid
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, auc, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def run_model(X_train, X_test, y_train, y_test, algorithm, max_depth=None):
    model = None
    is_classification = True

    try:
        if algorithm == 'logistic':
            model = LogisticRegression(max_iter=1000, random_state=42)
        elif algorithm == 'linear_regression':
            model = LinearRegression()
            is_classification = False
        elif algorithm == 'decision_tree':
            if is_classification:
                model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            else:
                model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        elif algorithm == 'random_forest':
            if is_classification:
                model = RandomForestClassifier(max_depth=max_depth, random_state=42)
            else:
                model = RandomForestRegressor(max_depth=max_depth, random_state=42)
        elif algorithm == 'knn':
            model = KNeighborsClassifier()
        elif algorithm == 'naive_bayes':
            model = GaussianNB()
        elif algorithm == 'svm':
            model = SVC(probability=True, random_state=42)
        elif algorithm == 'xgboost':
            import xgboost as xgb
            model = xgb.XGBClassifier(max_depth=max_depth, use_label_encoder=False, eval_metric='logloss', random_state=42)
        elif algorithm == 'lightgbm':
            import lightgbm as lgb
            model = lgb.LGBMClassifier(max_depth=max_depth, random_state=42)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize model: {e}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if is_classification:
        score = accuracy_score(y_test, y_pred)
    else:
        score = r2_score(y_test, y_pred)

    feature_importances = None
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        coefs = model.coef_
        feature_importances = coefs[0] if len(coefs.shape) > 1 else coefs

    roc_path = None
    if is_classification and len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")

            plot_filename = f"{uuid.uuid4()}_roc_curve.png"
            plot_path = os.path.join('static/plots', plot_filename)
            plt.savefig(plot_path)
            plt.close()
            roc_path = plot_path
        except Exception:
            roc_path = None

    return model, score, y_pred, feature_importances, roc_path

