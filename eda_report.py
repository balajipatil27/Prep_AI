import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import html

# Get base directory
BASE_DIR = Path(__file__).parent

def generate_eda_summary(df):
    """Generate EDA summary with properly escaped HTML"""
    try:
        # Generate HTML tables WITHOUT using pandas' to_html (it's causing issues)
        # Instead, create simple HTML tables
        
        # Head table
        head_html = "<table class='table table-striped table-bordered'><thead><tr>"
        for col in df.columns:
            head_html += f"<th>{html.escape(str(col))}</th>"
        head_html += "</tr></thead><tbody>"
        
        for _, row in df.head().iterrows():
            head_html += "<tr>"
            for val in row:
                head_html += f"<td>{html.escape(str(val))}</td>"
            head_html += "</tr>"
        head_html += "</tbody></table>"
        
        # Data Types table
        dtypes_df = df.dtypes.astype(str).reset_index()
        dtypes_df.columns = ['Column', 'Data Type']
        
        dtypes_html = "<table class='table table-striped table-bordered'><thead><tr><th>Column</th><th>Data Type</th></tr></thead><tbody>"
        for _, row in dtypes_df.iterrows():
            dtypes_html += f"<tr><td>{html.escape(str(row['Column']))}</td><td>{html.escape(str(row['Data Type']))}</td></tr>"
        dtypes_html += "</tbody></table>"
        
        # Missing values table
        missing_df = df.isnull().sum().reset_index()
        missing_df.columns = ['Column', 'Missing Values']
        
        missing_html = "<table class='table table-striped table-bordered'><thead><tr><th>Column</th><th>Missing Values</th></tr></thead><tbody>"
        for _, row in missing_df.iterrows():
            missing_html += f"<tr><td>{html.escape(str(row['Column']))}</td><td>{html.escape(str(row['Missing Values']))}</td></tr>"
        missing_html += "</tbody></table>"
        
        # Tail table
        tail_html = "<table class='table table-striped table-bordered'><thead><tr>"
        for col in df.columns:
            tail_html += f"<th>{html.escape(str(col))}</th>"
        tail_html += "</tr></thead><tbody>"
        
        for _, row in df.tail().iterrows():
            tail_html += "<tr>"
            for val in row:
                tail_html += f"<td>{html.escape(str(val))}</td>"
            tail_html += "</tr>"
        tail_html += "</tbody></table>"
        
        # Describe table (only for numeric columns)
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            describe_df = numeric_df.describe()
            describe_html = "<table class='table table-striped table-bordered'><thead><tr><th>Statistic</th>"
            for col in describe_df.columns:
                describe_html += f"<th>{html.escape(str(col))}</th>"
            describe_html += "</tr></thead><tbody>"
            
            for idx, row in describe_df.iterrows():
                describe_html += f"<tr><td>{html.escape(str(idx))}</td>"
                for col in describe_df.columns:
                    describe_html += f"<td>{html.escape(str(round(row[col], 4)))}</td>"
                describe_html += "</tr>"
            describe_html += "</tbody></table>"
        else:
            describe_html = "<p class='text-info'>No numeric columns for descriptive statistics</p>"
        
        return {
            'head': head_html,
            'shape': f"{df.shape[0]} rows × {df.shape[1]} columns",
            'dtypes': dtypes_html,
            'missing_values': missing_html,
            'describe': describe_html,
            'tail': tail_html
        }
    except Exception as e:
        print(f"Error in generate_eda_summary: {e}")
        import traceback
        traceback.print_exc()
        return {
            'error': str(e),
            'head': '<p class="text-danger">Error loading data preview</p>',
            'shape': 'Error',
            'dtypes': '<p class="text-danger">Error loading data types</p>',
            'missing_values': '<p class="text-danger">Error loading missing values</p>',
            'describe': '<p class="text-danger">Error loading statistics</p>',
            'tail': '<p class="text-danger">Error loading data tail</p>'
        }

def generate_correlation_plot(df, uid):
    """Generate correlation plot"""
    try:
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            print(f"Not enough numeric columns for correlation: {numeric_df.shape[1]}")
            return None
        
        plt.figure(figsize=(12, 10))
        corr = numeric_df.corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Plot heatmap
        sns.heatmap(corr, 
                   mask=mask,
                   annot=True, 
                   fmt=".2f", 
                   cmap='coolwarm',
                   center=0,
                   square=True,
                   linewidths=.5,
                   cbar_kws={"shrink": .8})
        
        plt.title('Correlation Heatmap', fontsize=16, pad=20)
        plt.tight_layout()
        
        # Save to plots directory
        plot_dir = BASE_DIR / 'static' / 'plots'
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        plot_filename = f'correlation_{uid}.png'
        plot_path = plot_dir / plot_filename
        plt.savefig(str(plot_path), bbox_inches='tight', dpi=100)
        plt.close()
        
        return plot_filename  # Return just the filename
    except Exception as e:
        print(f"Error generating correlation plot: {e}")
        import traceback
        traceback.print_exc()
        return None
