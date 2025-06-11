import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os
import matplotlib
matplotlib.use('Agg') 

def generate_graphs(df):
    graphs = []

    sns.set_style("whitegrid")
    
    # Graph 1: Distribution of a numerical column (Histogram/KDE)
    numerical_cols = df.select_dtypes(include=['number']).columns
    if len(numerical_cols) >= 1:
        col = numerical_cols[0]
        try:
            plt.figure(figsize=(8, 5)) 
            sns.histplot(df[col].dropna(), kde=True, bins=20, color='skyblue')
            plt.title(f'Distribution of {col}', fontsize=14)
            plt.xlabel(col, fontsize=10)
            plt.ylabel('Frequency', fontsize=10)
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png')
            plt.close()
            graphs.append({
                'title': f'Distribution of {col}',
                'data': base64.b64encode(buf.getvalue()).decode('utf-8')
            })
        except Exception as e:
            print(f"Error generating histogram for {col}: {e}")
            graphs.append({'title': f'Distribution of {col}', 'data': 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII='})

    # Graph 2: Scatter Plot of two numerical columns
    if len(numerical_cols) >= 2:
        col1, col2 = numerical_cols[0], numerical_cols[1]
        try:
            plt.figure(figsize=(8, 5))
            sns.scatterplot(x=df[col1], y=df[col2], alpha=0.7, color='orange')
            plt.title(f'{col1} vs {col2} (Scatter Plot)', fontsize=14)
            plt.xlabel(col1, fontsize=10)
            plt.ylabel(col2, fontsize=10)
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png')
            plt.close()
            graphs.append({
                'title': f'{col1} vs {col2} (Scatter Plot)',
                'data': base64.b64encode(buf.getvalue()).decode('utf-8')
            })
        except Exception as e:
            print(f"Error generating scatter plot for {col1}, {col2}: {e}")
            graphs.append({'title': f'{col1} vs {col2} (Scatter Plot)', 'data': 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII='})


    # Graph 3: Bar Chart for a categorical column
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) >= 1:
        col = categorical_cols[0]
        try:
            plt.figure(figsize=(8, 5)) 
            sns.countplot(y=df[col].dropna(), order=df[col].value_counts().index, palette='viridis')
            plt.title(f'Count of {col}', fontsize=14)
            plt.xlabel('Count', fontsize=10)
            plt.ylabel(col, fontsize=10)
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png')
            plt.close()
            graphs.append({
                'title': f'Count of {col}',
                'data': base64.b64encode(buf.getvalue()).decode('utf-8')
            })
        except Exception as e:
            print(f"Error generating bar chart for {col}: {e}")
            graphs.append({'title': f'Count of {col}', 'data': 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII='})

    # Graph 4: Box Plot for numerical distribution across categories
    if len(numerical_cols) >= 1 and len(categorical_cols) >= 1:
        num_col = numerical_cols[0]
        cat_col = categorical_cols[0]
        try:
            plt.figure(figsize=(8, 5)) 
            sns.boxplot(x=df[cat_col], y=df[num_col], palette='plasma')
            plt.title(f'Distribution of {num_col} by {cat_col}', fontsize=14)
            plt.xlabel(cat_col, fontsize=10)
            plt.ylabel(num_col, fontsize=10)
            plt.xticks(rotation=45, ha='right')
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png')
            plt.close()
            graphs.append({
                'title': f'Distribution of {num_col} by {cat_col}',
                'data': base64.b64encode(buf.getvalue()).decode('utf-8')
            })
        except Exception as e:
            print(f"Error generating box plot for {num_col} by {cat_col}: {e}")
            graphs.append({'title': f'Distribution of {num_col} by {cat_col}', 'data': 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII='})


    # Graph 5: Correlation Heatmap for numerical columns
    if len(numerical_cols) >= 2:
        try:
            plt.figure(figsize=(8, 5)) 
            corr_matrix = df[numerical_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
            plt.title('Correlation Heatmap of Numerical Features', fontsize=14)
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png')
            plt.close()
            graphs.append({
                'title': 'Correlation Heatmap',
                'data': base64.b64encode(buf.getvalue()).decode('utf-8')
            })
        except Exception as e:
            print(f"Error generating correlation heatmap: {e}")
            graphs.append({'title': 'Correlation Heatmap', 'data': 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII='})


    while len(graphs) < 5:
        graphs.append({
            'title': f'Placeholder Graph {len(graphs) + 1}',
            'data': 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=' # Base64 for a 1x1 transparent pixel
        })
        print(f"Added placeholder graph {len(graphs)}")

    return graphs

def generate_kpis(df):
    kpis = []
    numerical_cols = df.select_dtypes(include=['number']).columns

    if len(numerical_cols) > 0:
        # KPI 1: Average of the first numerical column
        col_avg = numerical_cols[0]
        try:
            avg_value = df[col_avg].mean()
            kpis.append({
                'label': f'Avg {col_avg}', 
                'value': f'{avg_value:,.1f}' 
            })
        except Exception as e:
            print(f"Error calculating average for {col_avg}: {e}")

    if len(numerical_cols) > 1:
        # KPI 2: Maximum of the second numerical column
        col_max = numerical_cols[1]
        try:
            max_value = df[col_max].max()
            kpis.append({
                'label': f'Max {col_max}', 
                'value': f'{max_value:,.0f}' 
            })
        except Exception as e:
            print(f"Error calculating max for {col_max}: {e}")
    
    # Add more KPIs if desired, for example:
    if len(numerical_cols) > 2:
        col_min = numerical_cols[2]
        try:
            min_value = df[col_min].min()
            kpis.append({
                'label': f'Min {col_min}', 
                'value': f'{min_value:,.0f}'
            })
        except Exception as e:
            print(f"Error calculating min for {col_min}: {e}")

    kpis.append({
        'label': 'Total Records',
        'value': f'{len(df):,}'
    })

    while len(kpis) < 3:
        kpis.append({
            'label': f'KPI Placeholder {len(kpis) + 1}',
            'value': 'N/A'
        })

    return kpis

def generate_key_statistics(df):
    """
    Generates a list of key statistics about the dataset.
    This will appear in the empty slot in the dashboard.
    """
    stats = []

    # Number of columns
    stats.append({'label': 'Total Columns', 'value': len(df.columns)})

    # Number of numerical columns
    numerical_cols = df.select_dtypes(include=['number']).columns
    stats.append({'label': 'Numerical Columns', 'value': len(numerical_cols)})

    # Number of categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    stats.append({'label': 'Categorical Columns', 'value': len(categorical_cols)})

    # Missing values count (overall)
    total_missing = df.isnull().sum().sum()
    stats.append({'label': 'Total Missing Values', 'value': total_missing})

    # Most frequent value in a categorical column (if available)
    if len(categorical_cols) > 0:
        col = categorical_cols[0]
        most_frequent_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A'
        stats.append({'label': f'Most Frequent {col}', 'value': most_frequent_value})
    else:
        stats.append({'label': 'Most Frequent (Category)', 'value': 'N/A'})
    
    # Mean of a numerical column (if available, for more insights)
    if len(numerical_cols) > 0:
        col = numerical_cols[0]
        mean_val = df[col].mean()
        stats.append({'label': f'Mean of {col}', 'value': f'{mean_val:,.2f}'})
    else:
        stats.append({'label': 'Mean (Numerical)', 'value': 'N/A'})

    return stats


def get_dashboard_data(filepath, filename):
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported file type. Please upload a CSV or XLSX file.")

        graphs = generate_graphs(df)
        kpis = generate_kpis(df)
        key_statistics = generate_key_statistics(df)
        
        return {
            'graphs': graphs,
            'kpis': kpis,
            'filename': filename,
            'key_statistics': key_statistics
        }
    except Exception as e:
        raise ValueError(f"Error processing file: {e}. Please ensure it's a valid CSV or XLSX.")

