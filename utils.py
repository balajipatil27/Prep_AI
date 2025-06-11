import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')  

import matplotlib.pyplot as plt


PLOT_FOLDER = 'static/plots'
os.makedirs(PLOT_FOLDER, exist_ok=True)

def generate_confusion_matrix_plot(y_true, y_pred, uid):
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')

        plot_path = os.path.join(PLOT_FOLDER, f'{uid}_confusion_matrix.png')
        plt.savefig(plot_path)
        plt.close()
        return plot_path
    except Exception as e:
        print(f"Error generating confusion matrix plot: {e}")
        return None

def generate_correlation_plot(df, uid):
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return None

    try:
        corr = numeric_df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar=True)
        plt.title('Correlation Matrix')

        plot_path = os.path.join(PLOT_FOLDER, f'{uid}_correlation_matrix.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        return plot_path
    except Exception as e:
        print(f"Error generating correlation matrix plot: {e}")
        return None

def generate_feature_importance_plot(importances, feature_names, uid):
    try:
        plt.figure(figsize=(10, 6))
        indices = np.argsort(importances)[::-1]
        sorted_features = [feature_names[i] for i in indices]
        sorted_importances = importances[indices]

        sns.barplot(x=sorted_importances, y=sorted_features, palette='viridis')
        plt.title('Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()

        plot_path = os.path.join(PLOT_FOLDER, f'{uid}_feature_importance.png')
        plt.savefig(plot_path)
        plt.close()
        return plot_path
    except Exception as e:
        print(f"Error generating feature importance plot: {e}")
        return None

def generate_shap_plot(model, X_train, uid, max_display=10):
    try:
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_train)

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_train, max_display=max_display, show=False)

        plt.legend([], [], frameon=False)

        plot_path = os.path.join('static/plots', f'{uid}_shap_summary.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        return plot_path

    except Exception as e:
        print(f"SHAP plot generation failed: {e}")
        return None
