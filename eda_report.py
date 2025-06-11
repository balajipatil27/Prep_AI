import pandas as pd

def generate_eda_summary(df: pd.DataFrame):
    eda = {}

    eda['head'] = df.head().to_html(classes='table table-striped table-hover', border=0)
    eda['tail'] = df.tail().to_html(classes='table table-striped table-hover', border=0)
    eda['shape'] = df.shape
    eda['dtypes'] = df.dtypes.apply(lambda x: x.name).to_dict()
    eda['missing_values'] = df.isnull().sum().to_dict()
    eda['unique_values'] = df.nunique().to_dict()

    numeric_desc = df.describe().transpose()
    eda['describe_numeric'] = numeric_desc.to_html(classes='table table-bordered table-sm', border=0)

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        cat_desc = df[categorical_cols].describe().transpose()
        eda['describe_categorical'] = cat_desc.to_html(classes='table table-bordered table-sm', border=0)
    else:
        eda['describe_categorical'] = "<p>No categorical columns present.</p>"

    return eda
