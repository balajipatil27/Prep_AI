import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_eda_summary(df):
    """Minimal EDA summary for your app"""
    return {
        'shape': f"{df.shape[0]} rows × {df.shape[1]} columns",
        'head': df.head().to_html(classes='table table-striped'),
        'tail': df.tail().to_html(classes='table table-striped'),
        'dtypes': df.dtypes.astype(str).to_frame('Data Type').to_html(classes='table table-striped'),
        'missing_values': df.isnull().sum().to_frame('Missing Values').to_html(classes='table table-striped'),
        'describe': df.describe().to_html(classes='table table-striped')
    }

def generate_correlation_plot(df, uid):
    """Minimal correlation plot for your app"""
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return None
            
        plt.figure(figsize=(10, 8))
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title(f'Correlation Heatmap - {uid}')
        
        plot_dir = '../static/plots'
        os.makedirs(plot_dir, exist_ok=True)
        
        plot_path = os.path.join(plot_dir, f'correlation_{uid}.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        
        return plot_path
    except Exception as e:
        print(f"Error in generate_correlation_plot: {e}")
        return None
