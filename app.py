from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
import os
import uuid
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import request, send_file, redirect, flash, session
from sklearn.preprocessing import LabelEncoder
import sys
import traceback
from pathlib import Path
import json

# ============ PATH CONFIGURATION ============
# Get the absolute path of the current directory
BASE_DIR = Path(__file__).parent.absolute()
print(f"BASE_DIR: {BASE_DIR}")

# Define all paths relative to BASE_DIR
UPLOAD_FOLDER = BASE_DIR / 'uploads'
PLOT_FOLDER = BASE_DIR / 'static' / 'plots'
PROCESSED_FOLDER = BASE_DIR / 'processed'
STATIC_FOLDER = BASE_DIR / 'static'
TEMPLATE_FOLDER = BASE_DIR / 'templates'

print(f"Template folder path: {TEMPLATE_FOLDER}")
print(f"Template folder exists: {TEMPLATE_FOLDER.exists()}")

# Add to system path for imports
sys.path.append(str(BASE_DIR))

# ============ CREATE DIRECTORIES ============
# Create all necessary directories
for folder in [UPLOAD_FOLDER, PLOT_FOLDER, PROCESSED_FOLDER, STATIC_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)
    print(f"Created/Verified directory: {folder}")

# ============ TRY TO IMPORT CUSTOM MODULES ============
def safe_import():
    """Safely import custom modules with fallback functions"""
    try:
        from preprocessing import preprocess_data
        print("✓ Imported preprocess_data")
    except ImportError:
        print("✗ Could not import preprocess_data, using fallback")
        from sklearn.model_selection import train_test_split
        def preprocess_data(df, target):
            """Fallback preprocessing function"""
            X = df.drop(columns=[target])
            y = df[target]
            return train_test_split(X, y, test_size=0.2, random_state=42)
    
    try:
        from model_runner import run_model
        print("✓ Imported run_model")
    except ImportError:
        print("✗ Could not import run_model, using fallback")
        from sklearn.linear_model import LogisticRegression
        def run_model(X_train, X_test, y_train, y_test, algorithm):
            """Fallback model function"""
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            return model, acc, y_pred
    
    try:
        from utils import generate_confusion_matrix_plot
        print("✓ Imported generate_confusion_matrix_plot")
    except ImportError:
        print("✗ Could not import generate_confusion_matrix_plot, using fallback")
        def generate_confusion_matrix_plot(y_test, y_pred, uid):
            """Fallback confusion matrix function"""
            try:
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                cm_filename = f'confusion_matrix_{uid}.png'
                cm_path = PLOT_FOLDER / cm_filename
                plt.title('Confusion Matrix')
                plt.tight_layout()
                plt.savefig(str(cm_path))
                plt.close()
                return str(cm_path)
            except Exception as e:
                print(f"Error generating confusion matrix: {e}")
                return None
    
    try:
        from eda_report import generate_eda_summary, generate_correlation_plot
        print("✓ Imported eda_report functions")
    except ImportError:
        print("✗ Could not import eda_report, using fallback")
        def generate_eda_summary(df):
            """Fallback EDA function"""
            try:
                return {
                    'head': df.head().to_html(classes='table table-striped', index=False),
                    'shape': f"{df.shape[0]} rows × {df.shape[1]} columns",
                    # 'dtypes': df.dtypes.astype(str).to_frame('Data Type').to_html(classes='table table-striped'),
                    # 'missing_values': df.isnull().sum().to_frame('Missing Values').to_html(classes='table table-striped'),
                    'describe': df.describe().to_html(classes='table table-striped'),
                    'tail': df.tail().to_html(classes='table table-striped', index=False)
                }
            except Exception as e:
                return {'error': str(e)}
        
        def generate_correlation_plot(df, uid):
            """Fallback correlation plot function"""
            try:
                numeric_df = df.select_dtypes(include=[np.number])
                if numeric_df.shape[1] < 2:
                    return None
                
                plt.figure(figsize=(10, 8))
                corr = numeric_df.corr()
                sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
                plt.title('Correlation Heatmap')
                plt.tight_layout()
                
                plot_filename = f'correlation_{uid}.png'
                plot_path = PLOT_FOLDER / plot_filename
                plt.savefig(str(plot_path), bbox_inches='tight')
                plt.close()
                
                return str(plot_path)
            except Exception as e:
                print(f"Error generating correlation plot: {e}")
                return None
    
    try:
        from dashboard import get_dashboard_data
        print("✓ Imported get_dashboard_data")
    except ImportError:
        print("✗ Could not import dashboard, using fallback")
        def get_dashboard_data(filepath, original_filename):
            """Fallback dashboard function"""
            try:
                if str(filepath).endswith('.csv'):
                    df = pd.read_csv(filepath)
                else:
                    df = pd.read_excel(filepath)
                
                return {
                    'filename': original_filename,
                    'shape': f"{df.shape[0]} rows × {df.shape[1]} columns",
                    'columns': list(df.columns),
                    'preview': df.head(10).to_html(classes='table table-striped', index=False),
                    'dtypes': df.dtypes.astype(str).to_frame('Data Type').to_html(classes='table table-striped'),
                    'missing_values': df.isnull().sum().to_frame('Missing Values').to_html(classes='table table-striped'),
                    'describe': df.describe().to_html(classes='table table-striped'),
                    'numeric_columns': list(df.select_dtypes(include=[np.number]).columns)
                }
            except Exception as e:
                raise ValueError(f"Error reading file: {str(e)}")
    
    # Return the functions
    return locals()

# Import all functions
imports = safe_import()
globals().update({k: v for k, v in imports.items() if callable(v)})

# ============ FLASK APP SETUP ============
app = Flask(__name__,
            template_folder=str(TEMPLATE_FOLDER),
            static_folder=str(STATIC_FOLDER))

app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

data_cache = {}

# ============ HELPER FUNCTIONS ============
def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_heatmap(df):
    """Generate correlation heatmap"""
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return None
        
        corr = numeric_df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
        heatmap_filename = 'correlation_heatmap.png'
        heatmap_path = PLOT_FOLDER / heatmap_filename
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(str(heatmap_path))
        plt.close()
        return heatmap_filename
    except Exception as e:
        print(f"Error generating heatmap: {e}")
        return None

def detect_and_convert_types(df, report):
    """Try to convert object columns to numeric"""
    for col in df.select_dtypes(include='object').columns:
        try:
            df[col] = pd.to_numeric(df[col])
            report.append(f"🔄 Converted column '{col}' from object to numeric.")
        except ValueError:
            continue
    return df

def extract_time_features(df, report):
    """Extract hour and minute from time columns"""
    for col in df.select_dtypes(include='object').columns:
        if df[col].astype(str).str.contains(':').any():
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
    """Cap outliers using IQR method"""
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
    """Encode categorical columns based on selected strategy"""
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
    """Main preprocessing function"""
    report = []
    ext = filename.rsplit('.', 1)[1].lower()
    
    try:
        if ext == 'xlsx':
            df = pd.read_excel(filepath)
        else:
            df = pd.read_csv(filepath)
    except Exception as e:
        report.append(f"❌ Error loading file: {e}")
        return "", report, "", "", ""

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

    df = df.replace({True: 1, False: 0})
    report.append("🔁 Converted boolean values True/False to 1/0.")

    report.append(f"✅ Final dataset shape: {df.shape}")

    heatmap_filename = generate_heatmap(df)
    if heatmap_filename:
        report.append("📊 Correlation heatmap generated.")
        heatmap_url = url_for('static', filename=f'plots/{heatmap_filename}')
    else:
        report.append("❌ Could not generate correlation heatmap (no numeric columns).")
        heatmap_url = ""

    cleaned_filename = 'cleaned_' + filename
    cleaned_path = PROCESSED_FOLDER / cleaned_filename
    
    try:
        if ext == 'xlsx':
            df.to_excel(str(cleaned_path), index=False)
        else:
            df.to_csv(str(cleaned_path), index=False)
    except Exception as e:
        report.append(f"❌ Error saving cleaned file: {e}")
        cleaned_path = ""

    report_path = PROCESSED_FOLDER / 'preprocessing_log.txt'
    try:
        with open(str(report_path), 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
    except Exception as e:
        report.append(f"❌ Error saving report: {e}")
        report_path = ""

    preview_html = df.head().to_html(classes='table table-striped', index=False) if not df.empty else "<p>No data available</p>"
    
    return preview_html, report, str(cleaned_path) if cleaned_path else "", str(report_path) if report_path else "", heatmap_url

# ============ ROUTES ============
@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/eda')
def eda():
    """EDA page"""
    return render_template('eda.html')

@app.route('/pre')
def pre():
    """Preprocessing page"""
    return render_template('pre.html')

@app.route('/aboutus')
def aboutus():
    """About page"""
    return render_template('about.html')

@app.route('/dash')
def dash():
    """Dashboard page"""
    return render_template('dashboardind.html')

@app.route('/upload_preprocess', methods=['POST'])
def upload_preprocess():
    """Handle file upload for preprocessing"""
    if 'file' not in request.files:
        flash("No file part.")
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash("No selected file.")
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filepath = UPLOAD_FOLDER / file.filename
        file.save(str(filepath))
        ext = file.filename.rsplit('.', 1)[1].lower()
        
        try:
            if ext == 'xlsx':
                df = pd.read_excel(str(filepath))
            else:
                df = pd.read_csv(str(filepath))
        except Exception as e:
            flash(f"Error reading file: {str(e)}")
            return redirect(url_for('pre'))
            
        categorical_columns = df.select_dtypes(include='object').columns.tolist()

        session['filename'] = file.filename
        column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
        return render_template('encoding_options.html', 
                             categorical_columns=categorical_columns, 
                             column_types=column_types, 
                             filename=file.filename)

    flash("Invalid file format.")
    return redirect(url_for('pre'))

@app.route('/preprocess', methods=['POST'])
def preprocess():
    """Handle preprocessing with encoding strategies"""
    filename = session.get('filename')
    if not filename:
        flash("Session expired or no file uploaded.")
        return redirect(url_for('pre'))

    filepath = UPLOAD_FOLDER / filename
    if not filepath.exists():
        flash("File not found. Please upload again.")
        return redirect(url_for('pre'))

    ext = filename.rsplit('.', 1)[1].lower()
    try:
        if ext == 'xlsx':
            df = pd.read_excel(str(filepath))
        else:
            df = pd.read_csv(str(filepath))
    except Exception as e:
        flash(f"Error reading file: {str(e)}")
        return redirect(url_for('pre'))
        
    categorical_columns = df.select_dtypes(include='object').columns.tolist()

    strategy_dict = {}
    for col in categorical_columns:
        strategy = request.form.get(f"encoding_strategy_{col}", "ignore")
        strategy_dict[col] = strategy

    preview_html, report, cleaned_path, report_path, heatmap_url = preprocess_dataset(
        str(filepath), filename, strategy_dict
    )
    
    return render_template('result.html', 
                         table=preview_html, 
                         report=report,
                         cleaned_filename=os.path.basename(cleaned_path) if cleaned_path else '',
                         log_filename=os.path.basename(report_path) if report_path else '',
                         heatmap_image=heatmap_url)

@app.route('/download/<filename>')
def download_file(filename):
    """Download processed files"""
    path = PROCESSED_FOLDER / filename
    if path.exists():
        return send_file(str(path), as_attachment=True)
    else:
        flash("File not found.")
        return redirect(url_for('pre'))

@app.route('/upload', methods=['POST'])
def upload():
    """Handle file upload for EDA"""
    try:
        file = request.files.get('dataset')
        if not file:
            return jsonify({"error": "No file uploaded"}), 400
            
        uid = str(uuid.uuid4())
        filename = uid + "_" + file.filename
        filepath = UPLOAD_FOLDER / filename
        file.save(str(filepath))

        # Read file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(str(filepath))
        else:
            df = pd.read_excel(str(filepath))
            
        data_cache[uid] = df

        # Generate EDA
        eda_data = generate_eda_summary(df)
        
        # Generate correlation plot
        corr_path = generate_correlation_plot(df, uid)
        
        # Clean up uploaded file
        if filepath.exists():
            filepath.unlink()

        return jsonify({
            "uid": uid,
            "columns": list(df.columns),
            "preview": eda_data.get('head', ''),
            "eda": {
                "shape": eda_data.get('shape', ''),
                "dtypes": eda_data.get('dtypes', ''),
                "missing": eda_data.get('missing_values', ''),
                "describe": eda_data.get('describe', ''),
                "tail": eda_data.get('tail', ''),
                "corr_path": f"/static/plots/correlation_{uid}.png" if corr_path else None
            }
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/train', methods=['POST'])
def train():
    """Train model endpoint"""
    try:
        uid = request.form.get('uid')
        target = request.form.get('target')
        algorithm = request.form.get('algorithm')

        if not all([uid, target, algorithm]):
            return jsonify({"error": "Missing parameters"}), 400

        df = data_cache.get(uid)
        if df is None:
            return jsonify({"error": "Data not found"}), 400

        # Preprocess data
        X_train, X_test, y_train, y_test = preprocess_data(df, target)
        
        # Run model
        model, acc, y_pred = run_model(X_train, X_test, y_train, y_test, algorithm)
        
        # Generate confusion matrix
        cm_path = generate_confusion_matrix_plot(y_test, y_pred, uid)
        
        # Prepare response
        response = {
            "accuracy": round(acc * 100, 2),
            "algorithm": algorithm,
            "target": target,
            "data_shape": f"{df.shape[0]} rows, {df.shape[1]} columns"
        }
        
        if cm_path:
            cm_filename = os.path.basename(cm_path)
            response["confusion_matrix"] = url_for('static', filename=f'plots/{cm_filename}')
        
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/static/plots/<filename>')
def serve_plot(filename):
    """Serve plot files"""
    try:
        return send_from_directory(str(PLOT_FOLDER), filename)
    except Exception as e:
        print(f"Error serving plot {filename}: {e}")
        return "Plot not found", 404

@app.route('/uploaddash', methods=['POST'])
def upload_file():
    """Handle dashboard file upload"""
    if 'file' not in request.files:
        return render_template('dashboardind.html', error="No file part")
        
    file = request.files['file']
    if file.filename == '':
        return render_template('dashboardind.html', error="No selected file")
    
    if file and allowed_file(file.filename):
        original_filename = file.filename
        unique_filename = str(uuid.uuid4()) + "_" + original_filename
        filepath = UPLOAD_FOLDER / unique_filename
        file.save(str(filepath))

        try:
            dashboard_data = get_dashboard_data(str(filepath), original_filename)
            return render_template('dsahresult.html', **dashboard_data)
        except ValueError as e:
            return render_template('dashboardind.html', error=str(e))
        except Exception as e:
            return render_template('dashboardind.html', error=f"Error processing file: {str(e)}")
        finally:
            if filepath.exists():
                filepath.unlink()
    else:
        return render_template('dashboardind.html', error="Invalid file type. Please upload a CSV or XLSX file.")

# ============ ERROR HANDLERS ============
@app.errorhandler(404)
def not_found_error(error):
    """404 error handler"""
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    """500 error handler"""
    return render_template('error.html', error="Internal server error"), 500

@app.errorhandler(413)
def too_large(error):
    """File too large error"""
    return render_template('error.html', error="File is too large (max 16MB)"), 413

# ============ CREATE BASIC ERROR TEMPLATE ============
def create_error_template():
    """Create basic error template if it doesn't exist"""
    error_template_path = TEMPLATE_FOLDER / 'error.html'
    if not error_template_path.exists():
        error_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Error - Prep AI</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .error-container {
            background: white;
            border-radius: 15px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 600px;
            width: 100%;
        }
        .error-code {
            font-size: 72px;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 20px;
        }
        .error-message {
            font-size: 18px;
            color: #666;
            margin-bottom: 30px;
        }
        .btn-home {
            background: #667eea;
            color: white;
            padding: 12px 30px;
            border-radius: 50px;
            text-decoration: none;
            display: inline-block;
            transition: all 0.3s;
        }
        .btn-home:hover {
            background: #5a67d8;
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }
    </style>
</head>
<body>
    <div class="error-container text-center">
        <div class="error-code">{% if error == 'Page not found' %}404{% else %}500{% endif %}</div>
        <h1 class="mb-4">{{ error }}</h1>
        <p class="error-message">
            {% if error == 'Page not found' %}
                The page you're looking for doesn't exist or has been moved.
            {% else %}
                Something went wrong on our end. Please try again later.
            {% endif %}
        </p>
        <a href="/" class="btn-home">Go to Homepage</a>
    </div>
</body>
</html>"""
        with open(str(error_template_path), 'w') as f:
            f.write(error_html)
        print(f"Created error template at: {error_template_path}")

# ============ STARTUP CHECKS ============
def check_templates():
    """Check if all required templates exist"""
    required_templates = [
        'index.html', 'pre.html', 'eda.html', 'about.html',
        'dashboardind.html', 'encoding_options.html', 'result.html',
        'dsahresult.html'
    ]
    
    missing = []
    for template in required_templates:
        path = TEMPLATE_FOLDER / template
        if not path.exists():
            missing.append(template)
    
    if missing:
        print(f"⚠️ Warning: Missing templates: {missing}")
        print(f"Template folder: {TEMPLATE_FOLDER}")
    
    # Create error template if it doesn't exist
    create_error_template()
    
    return len(missing) == 0

# ============ MAIN ============
if __name__ == '__main__':
    # Check templates
    templates_ok = check_templates()
    
    if not templates_ok:
        print("⚠️ Some templates are missing. The app may not work correctly.")
    
    # Print debug info
    print("\n" + "="*50)
    print("PREP AI - Flask Application")
    print("="*50)
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Static folder: {STATIC_FOLDER}")
    print(f"Template folder: {TEMPLATE_FOLDER}")
    print(f"Plot folder: {PLOT_FOLDER}")
    print("="*50)
    print("Server starting on http://0.0.0.0:5000")
    print("="*50 + "\n")
    
    # Run app
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(
        debug=debug_mode,
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        threaded=True
    )

