import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

def preprocess_data(df: pd.DataFrame, target_column: str):
    """
    Preprocess the dataset:
    - Drop rows with missing target values
    - Split into train and test sets
    - Encode target labels with LabelEncoder fitted on train only;
      unseen test labels mapped to -1
    - Encode categorical features with LabelEncoder (fit on train only);
      unseen categories mapped to -1
    - Impute missing numeric values with mean, categorical with mode
    """
    df = df.copy()

    # Drop rows where target is missing
    df = df.dropna(subset=[target_column])

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Stratify if target is categorical with <=10 classes
    stratify = y if (y.dtype == 'object' or str(y.dtype).startswith('category')) and (y.nunique() <= 10) else None

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    # Impute numeric columns
    numeric_cols = X_train.select_dtypes(include=['number']).columns
    num_imputer = SimpleImputer(strategy='mean')
    X_train[numeric_cols] = num_imputer.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = num_imputer.transform(X_test[numeric_cols])

    # Impute categorical columns
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

    # Encode categorical features with LabelEncoder fitted on train only
    for col in cat_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = X_test[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)

    # Encode target labels with LabelEncoder fitted on train only
    if y_train.dtype == 'object' or str(y_train.dtype).startswith('category'):
        le_target = LabelEncoder()
        y_train_enc = le_target.fit_transform(y_train.astype(str))

        known_classes = set(le_target.classes_)
        y_test_enc = y_test.astype(str).map(lambda x: le_target.transform([x])[0] if x in known_classes else -1)
        y_test_enc = y_test_enc.astype(int)
    else:
        y_train_enc = y_train
        y_test_enc = y_test

    return X_train, X_test, y_train_enc, y_test_enc



