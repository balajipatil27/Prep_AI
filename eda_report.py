import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

def generate_eda_summary(df):
    """Minimal EDA summary for your app"""
    try:
        # Convert DataFrame to HTML with proper escaping
        head_html = df.head().to_html(
            classes='table table-striped table-bordered', 
            index=False,
            escape=True
        )
        
        tail_html = df.tail().to_html(
            classes='table table-striped table-bordered', 
            index=False,
            escape=True
        )
        
        dtypes_html = df.dtypes.astype(str).to_frame('Data Type').to_html(
            classes='table table-striped table-bordered',
            escape=True
        )
        
        missing_html = df.isnull().sum().to_frame('Missing Values').to_html(
            classes='table table-striped table-bordered',
            escape=True
        )
        
        describe_html = df.describe().to_html(
            classes='table table-striped table-bordered',
            escape=True
        )
        
        return {
            'shape': f"{df.shape[0]} rows × {df.shape[1]} columns",
            'head': head_html,
            'tail': tail_html,
            'dtypes': dtypes_html,
            'missing_values': missing_html,
            'describe': describe_html
        }
    except Exception as e:
        print(f"Error in generate_eda_summary: {e}")
        # Return safe fallback
        return {
            'shape': 'Error processing data',
            'head': '<p class="text-danger">Error loading preview</p>',
            'tail': '<p class="text-danger">Error loading preview</p>',
            'dtypes': '<p class="text-danger">Error loading data types</p>',
            'missing_values': '<p class="text-danger">Error loading missing values</p>',
            'describe': '<p class="text-danger">Error loading statistics</p>'
        }

def generate_correlation_plot(df, uid):
    """Minimal correlation plot for your app"""
    try:
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Need at least 2 numeric columns for correlation
        if numeric_df.shape[1] < 2:
            print(f"Not enough numeric columns for correlation. Found {numeric_df.shape[1]} columns.")
            return None
        
        # Create plot
        plt.figure(figsize=(10, 8))
        corr = numeric_df.corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Create heatmap
        sns.heatmap(corr, 
                   mask=mask,
                   annot=True, 
                   fmt=".2f", 
                   cmap='coolwarm',
                   center=0,
                   square=True,
                   linewidths=.5,
                   cbar_kws={"shrink": .8})
        
        plt.title(f'Correlation Heatmap')
        plt.tight_layout()
        
        # Ensure plots directory exists
        plot_dir = project_root / 'static' / 'plots'
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Save plot
        plot_path = plot_dir / f'correlation_{uid}.png'
        plt.savefig(plot_path, bbox_inches='tight', dpi=100)
        plt.close()
        
        # Return relative path for web
        return f'static/plots/correlation_{uid}.png'
        
    except Exception as e:
        print(f"Error in generate_correlation_plot: {e}")
        import traceback
        traceback.print_exc()
        return None
