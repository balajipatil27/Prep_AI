import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Get base directory
BASE_DIR = Path(__file__).parent

def generate_eda_summary(df):
    """Generate EDA summary"""
    try:
        return {
            'head': df.head().to_html(classes='table table-striped', index=False),
            'shape': f"{df.shape[0]} rows × {df.shape[1]} columns",
            'dtypes': df.dtypes.astype(str).to_frame('Data Type').to_html(classes='table table-striped'),
            'missing_values': df.isnull().sum().to_frame('Missing Values').to_html(classes='table table-striped'),
            'describe': df.describe().to_html(classes='table table-striped'),
            'tail': df.tail().to_html(classes='table table-striped', index=False)
        }
    except Exception as e:
        print(f"Error in generate_eda_summary: {e}")
        return {'error': str(e)}

def generate_correlation_plot(df, uid):
    """Generate correlation plot"""
    try:
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            print(f"Not enough numeric columns for correlation: {numeric_df.shape[1]}")
            return None
        
        plt.figure(figsize=(10, 8))
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        
        # Save to plots directory
        plot_dir = BASE_DIR / 'static' / 'plots'
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        plot_path = plot_dir / f'correlation_{uid}.png'
        plt.savefig(str(plot_path), bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    except Exception as e:
        print(f"Error generating correlation plot: {e}")
        return None
