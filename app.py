from flask import Flask, request, jsonify, render_template, send_from_directory, url_for, flash, session, send_file, redirect
import os
import uuid
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Essential Flask setup remains
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PLOT_FOLDER = 'static/plots'
REPORT_FOLDER = 'static/reports'
app.secret_key = 'your_secret_key'

PROCESSED_FOLDER = 'processed'
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

data_cache = {} 

# Home Page
@app.route('/eda')
def eda():
    return render_template('PlayML.html')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pre')
def pre():
    return render_template('pre.html')

@app.route('/aboutus')
def aboutus():
    return render_template('about.html')

@app.route('/dash')
def dash():
    return render_template('dashboard.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_heatmap(df):
    corr = df.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    heatmap_path = os.path.join(STATIC_FOLDER, 'correlation_heatmap.png')
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()
    return heatmap_path

def detect_and_convert_types(df, report):
    for col in df.select_dtypes(include='object').columns:
        try:
            df[col] = pd.to_numeric(df[col])
            report.append(f"🔄 Converted column '{col}' from object to numeric.")
        except ValueError:
            continue
    return df

def extract_time_features(df, report):
    for col in df.select_dtypes(include='object').columns:
        if df[col].str.contains(':').any():
            try:
                parsed_times = pd.to_datetime(df[col], errors='coerce')
                df[f'{col}_hour'] = parsed_times.dt.hour
                df[f'{col}_minute'] = parsed_times.dt.minute
                df.drop(columns=[col], inplace=True)
                report.append(f"🕒 Extracted time features from '{col}' → '{col}_hour', '{col}_minute'")
            except Exception as e:
                report.append(f"❗ Failed to parse time column '{col}': {e}")
    return df

def handle_outliers(df, report):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if not outliers.empty:
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
            report.append(f"⚠️ Capped outliers in '{col}' using IQR method.")
    return df

def encode_categorical(df, strategy_dict, report):
    for col, method in strategy_dict.items():
        if method == 'ignore':
            report.append(f"⏭️ Skipped encoding for '{col}'.")
            continue
        elif method == 'onehot':
            try:
                df = pd.get_dummies(df, columns=[col], prefix=col)
                report.append(f"🎯 Applied One-Hot Encoding to '{col}'.")
            except Exception as e:
                report.append(f"❗ Failed One-Hot Encoding for '{col}': {e}")
        elif method == 'label':
            try:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                report.append(f"🏷️ Applied Label Encoding to '{col}'.")
            except Exception as e:
                report.append(f"❗ Failed Label Encoding for '{col}': {e}")
    return df

def preprocess_dataset(filepath, filename, strategy_dict):
    report = []
    ext = filename.rsplit('.', 1)[1].lower()
    df = pd.read_excel(filepath) if ext == 'xlsx' else pd.read_csv(filepath)

    report.append(f"✔️ Loaded dataset with shape {df.shape}.")

    df = detect_and_convert_types(df, report)
    df = extract_time_features(df, report)

    null_percent = df.isnull().mean()
    cols_to_drop = null_percent[null_percent > 0.5].index.tolist()
    df.drop(columns=cols_to_drop, inplace=True)
    if cols_to_drop:
        report.append(f"❌ Dropped columns with >50% nulls: {cols_to_drop}")
    else:
        report.append("✅ No columns dropped (nulls < 50%).")

    for col in df.columns:
        if df[col].isnull().any() and df[col].dtype in [np.float64, np.int64]:
            mean = df[col].mean()
            median = df[col].median()
            mode = df[col].mode()[0] if not df[col].mode().empty else mean
            fill_value = round((mean + median + mode) / 3, 2)
            df[col].fillna(fill_value, inplace=True)
            report.append(f"🧪 Filled nulls in '{col}' with avg of mean={mean:.2f}, median={median:.2f}, mode={mode:.2f} → {fill_value}")

    before = df.shape[0]
    df.drop_duplicates(inplace=True)
    after = df.shape[0]
    removed = before - after
    if removed:
        report.append(f"🧹 Removed {removed} duplicate rows.")
    else:
        report.append("✅ No duplicate rows found.")

    df = handle_outliers(df, report)
    df = encode_categorical(df, strategy_dict, report)

    # Convert True/False to 1/0
    df = df.replace({True: 1, False: 0})
    report.append("🔁 Converted boolean values True/False to 1/0.")

    report.append(f"✅ Final dataset shape: {df.shape}")

    heatmap_path = generate_heatmap(df)
    report.append("📊 Correlation heatmap generated.")

    cleaned_path = os.path.join(PROCESSED_FOLDER, 'cleaned_' + filename)
    if ext == 'xlsx':
        df.to_excel(cleaned_path, index=False)
    else:
        df.to_csv(cleaned_path, index=False)

    report_path = os.path.join(PROCESSED_FOLDER, 'preprocessing_log.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    return df.head().to_html(classes='table table-striped'), report, cleaned_path, report_path, heatmap_path

@app.route('/upload_preprocess', methods=['POST'])
def upload_preprocess():
    if 'file' not in request.files:
        flash("No file part.")
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash("No selected file.")
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        ext = file.filename.rsplit('.', 1)[1].lower()
        df = pd.read_excel(filepath) if ext == 'xlsx' else pd.read_csv(filepath)
        categorical_columns = df.select_dtypes(include='object').columns.tolist()

        session['filename'] = file.filename
        column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
        return render_template('encoding_options.html', categorical_columns=categorical_columns, column_types=column_types, filename=file.filename)

    flash("Invalid file format.")
    return redirect(url_for('pre'))

@app.route('/preprocess', methods=['POST'])
def preprocess():
    filename = session.get('filename')
    if not filename:
        flash("Session expired or no file uploaded.")
        return redirect(url_for('pre'))

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    ext = filename.rsplit('.', 1)[1].lower()
    df = pd.read_excel(filepath) if ext == 'xlsx' else pd.read_csv(filepath)
    categorical_columns = df.select_dtypes(include='object').columns.tolist()

    strategy_dict = {}
    for col in categorical_columns:
        strategy = request.form.get(f"encoding_strategy_{col}", "ignore")
        strategy_dict[col] = strategy

    preview_html, report, cleaned_path, report_path, heatmap_path = preprocess_dataset(filepath, filename, strategy_dict)
    return render_template('result.html', table=preview_html, report=report,
                           cleaned_filename=os.path.basename(cleaned_path),
                           log_filename=os.path.basename(report_path),
                           heatmap_image=url_for('static', filename='correlation_heatmap.png'))

@app.route('/download/<filename>')
def download_file(filename):
    path = os.path.join(PROCESSED_FOLDER, filename)
    return send_file(path, as_attachment=True)

def suggest_algorithms(df, target_col):
    # Lazy import for optional dependencies
    try:
        import xgboost
        xgboost_available = True
    except ImportError:
        xgboost_available = False
    
    try:
        import lightgbm
        lightgbm_available = True
    except ImportError:
        lightgbm_available = False

    target_dtype = df[target_col].dtype
    unique_values = df[target_col].nunique()
    feature_types = df.drop(columns=[target_col]).dtypes

    has_categorical_features = any(
        dt == 'object' or str(dt).startswith('category') for dt in feature_types
    )
    has_numeric_features = any(
        pd.api.types.is_numeric_dtype(dt) for dt in feature_types
    )

    algorithms = []

    # Regression task
    if pd.api.types.is_numeric_dtype(target_dtype):
        algorithms = [
            {"value": "linear_regression", "label": "Linear Regression"},
            {"value": "decision_tree", "label": "Decision Tree"},
            {"value": "random_forest", "label": "Random Forest"},
        ]
        
        if xgboost_available:
            algorithms.append({"value": "xgboost", "label": "XGBoost"})
        
        if lightgbm_available:
            algorithms.append({"value": "lightgbm", "label": "LightGBM"})

        if not has_numeric_features:
            algorithms = [a for a in algorithms if a["value"] not in ("knn", "svm")]

    # Classification task
    elif target_dtype == 'object' or target_dtype.name == 'category':
        if unique_values == 2:
            algorithms = [
                {"value": "logistic", "label": "Logistic Regression"},
                {"value": "decision_tree", "label": "Decision Tree"},
                {"value": "random_forest", "label": "Random Forest"},
                {"value": "svm", "label": "SVM"},
                {"value": "naive_bayes", "label": "Naive Bayes"},
                {"value": "knn", "label": "KNN"},
            ]
            
            if xgboost_available:
                algorithms.append({"value": "xgboost", "label": "XGBoost"})
            
            if lightgbm_available:
                algorithms.append({"value": "lightgbm", "label": "LightGBM"})

            if not has_categorical_features:
                algorithms = [a for a in algorithms if a["value"] != "naive_bayes"]
        else:
            algorithms = [
                {"value": "decision_tree", "label": "Decision Tree"},
                {"value": "random_forest", "label": "Random Forest"},
                {"value": "svm", "label": "SVM"},
                {"value": "knn", "label": "KNN"},
            ]
            
            if xgboost_available:
                algorithms.append({"value": "xgboost", "label": "XGBoost"})
            
            if lightgbm_available:
                algorithms.append({"value": "lightgbm", "label": "LightGBM"})

    else:
        algorithms = [
            {"value": "decision_tree", "label": "Decision Tree"},
            {"value": "random_forest", "label": "Random Forest"},
        ]

    return algorithms

@app.route('/upload', methods=['POST'])
def upload():
    # Lazy import for EDA module
    from eda_report import generate_eda_summary
    from utils import generate_correlation_plot
    
    file = request.files.get('dataset')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    allowed_ext = {'csv', 'xls', 'xlsx'}
    filename = file.filename
    if '.' not in filename or filename.rsplit('.', 1)[1].lower() not in allowed_ext:
        return jsonify({"error": "Unsupported file type. Please upload CSV or Excel files."}), 400

    uid = str(uuid.uuid4())
    safe_filename = uid + "_" + filename
    filepath = os.path.join(UPLOAD_FOLDER, safe_filename)

    try:
        file.save(filepath)
    except Exception as e:
        return jsonify({"error": f"Failed to save file: {str(e)}"}), 500

    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
    except Exception as e:
        return jsonify({"error": f"Failed to read file: {str(e)}"}), 400

    data_cache[uid] = df

    try:
        eda = generate_eda_summary(df)
        corr_path = generate_correlation_plot(df, uid)
    except Exception as e:
        return jsonify({"error": f"EDA generation failed: {str(e)}"}), 500

    return jsonify({
        "uid": uid,
        "columns": list(df.columns),
        "preview": eda.get("head", ""),
        "eda": {
            "shape": eda.get("shape", (0, 0)),
            "dtypes": eda.get("dtypes", {}),
            "missing": eda.get("missing_values", {}),
            "unique_values": eda.get("unique_values", {}),
            "describe_numeric": eda.get("describe_numeric", ""),
            "describe_categorical": eda.get("describe_categorical", ""),
            "tail": eda.get("tail", ""),
            "corr_path": f"/{corr_path}" if corr_path else None
        }
    })

@app.route('/suggest_algorithms', methods=['POST'])
def suggest_algorithms_route():
    uid = request.form.get('uid')
    target = request.form.get('target')

    if not uid or not target:
        return jsonify({"error": "Missing uid or target"}), 400

    df = data_cache.get(uid)
    if df is None:
        return jsonify({"error": "Dataset not found"}), 400

    if target not in df.columns:
        return jsonify({"error": "Target column not found in dataset"}), 400

    try:
        algorithms = suggest_algorithms(df, target)
    except Exception as e:
        return jsonify({"error": f"Algorithm suggestion failed: {str(e)}"}), 500

    return jsonify({"suggested_algorithms": algorithms})

@app.route('/train', methods=['POST'])
def train():
    # Lazy import ML modules
    from preprocessing import preprocess_data
    from model_runner import run_model
    from utils import (
        generate_confusion_matrix_plot,
        generate_feature_importance_plot,
        generate_shap_plot
    )
    
    uid = request.form.get('uid')
    target = request.form.get('target')
    algorithm = request.form.get('algorithm')
    max_depth = request.form.get('max_depth', type=int, default=None)

    if not uid or not target or not algorithm:
        return jsonify({"error": "Missing required parameters"}), 400

    df = data_cache.get(uid)
    if df is None:
        return jsonify({"error": "Data not found"}), 400

    try:
        X_train, X_test, y_train, y_test = preprocess_data(df, target)
    except Exception as e:
        return jsonify({"error": f"Preprocessing failed: {str(e)}"}), 500

    try:
        model, acc, y_pred, feature_importances, roc_path = run_model(
            X_train, X_test, y_train, y_test, algorithm, max_depth=max_depth
        )
    except Exception as e:
        return jsonify({"error": f"Model training failed: {str(e)}"}), 500

    try:
        cm_path = generate_confusion_matrix_plot(y_test, y_pred, uid)
    except Exception:
        cm_path = None

    fi_plot_path = None
    if feature_importances is not None:
        try:
            fi_plot_path = generate_feature_importance_plot(feature_importances, X_train.columns, uid)
        except Exception:
            fi_plot_path = None

    shap_path = None
    try:
        shap_path = generate_shap_plot(model, X_train, uid)
    except Exception:
        shap_path = None

    return jsonify({
        "accuracy": round(acc * 100, 2),
        "confusion_matrix": url_for('serve_plot', filename=os.path.basename(cm_path)) if cm_path else None,
        "feature_importance": url_for('serve_plot', filename=os.path.basename(fi_plot_path)) if fi_plot_path else None,
        "roc_curve": url_for('serve_plot', filename=os.path.basename(roc_path)) if roc_path else None,
        "shap_summary": url_for('serve_plot', filename=os.path.basename(shap_path)) if shap_path else None
    })

@app.route('/static/plots/<filename>')
def serve_plot(filename):
    return send_from_directory(PLOT_FOLDER, filename)

#dashboard
@app.route('/uploaddash', methods=['POST'])
def upload_file():
    # Lazy import dashboard module
    from dashboard import get_dashboard_data
    
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Generate a unique filename to avoid overwriting issues
        original_filename = file.filename
        unique_filename = str(uuid.uuid4()) + "_" + original_filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        try:
            # Call the modularized function to get dashboard data
            # Pass the original filename to display in the dashboard header
            dashboard_data = get_dashboard_data(filepath, original_filename)
            return render_template('dashresult.html', **dashboard_data) # Unpack dict
        except ValueError as e:
            # Handle specific ValueErrors from get_dashboard_data (e.g., unsupported file type)
            return render_template('dashboard.html', error=str(e))
        except Exception as e:
            # Catch any other unexpected errors during processing
            return render_template('dashboard.html', error=f"An unexpected error occurred during file processing: {e}. Please check your file.")
        finally:
            # Clean up the uploaded file after processing
            if os.path.exists(filepath):
                os.remove(filepath)
    else:
        return render_template('dashboard.html', error="Invalid file type. Please upload a CSV or XLSX file.")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    # For production, use these settings
    if not debug_mode:
        # Production settings
        app.config['TEMPLATES_AUTO_RELOAD'] = False
        app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 300
        
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
