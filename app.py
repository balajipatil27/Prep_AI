from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
import os
import uuid
import pandas as pd
from preprocessing import preprocess_data
from model_runner import run_model
from utils import generate_confusion_matrix_plot
from eda_report import generate_eda_summary, generate_correlation_plot
from dashboard import get_dashboard_data
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from flask import request, send_file, redirect, flash, session
from sklearn.preprocessing import LabelEncoder

# Get the absolute path to the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define template and static folders relative to BASE_DIR
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

# Check if directories exist, create them if they don't
os.makedirs(TEMPLATE_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

print(f"Template directory: {TEMPLATE_DIR}")
print(f"Static directory: {STATIC_DIR}")
print(f"Template directory exists: {os.path.exists(TEMPLATE_DIR)}")
print(f"Static directory exists: {os.path.exists(STATIC_DIR)}")

# List files in template directory (for debugging)
if os.path.exists(TEMPLATE_DIR):
    print(f"Files in template directory: {os.listdir(TEMPLATE_DIR)}")

app = Flask(__name__, 
            template_folder=TEMPLATE_DIR,
            static_folder=STATIC_DIR)

UPLOAD_FOLDER = 'uploads'
PLOT_FOLDER = os.path.join(STATIC_DIR, 'plots')

app.secret_key = 'your_secret_key_here_change_me'

PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

data_cache = {} 
file_access_times = {}  # Track when files were last accessed
MAX_CACHE_AGE_HOURS = 1  # Clean up files older than 1 hour
MAX_CACHE_SIZE = 10      # Maximum number of files to keep in cache

# ==================== CLEANUP FUNCTIONS ====================

def cleanup_old_files():
    """Remove old files from uploads, processed folders, and clear old cache entries"""
    current_time = time.time()
    cutoff_time = current_time - (MAX_CACHE_AGE_HOURS * 3600)
    
    # Clean up UPLOAD_FOLDER
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.isfile(file_path):
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > cutoff_time:
                try:
                    os.remove(file_path)
                    print(f"Cleaned up old file: {filename}")
                except Exception as e:
                    print(f"Error removing {filename}: {e}")
    
    # Clean up PROCESSED_FOLDER (keep only files from current session)
    for filename in os.listdir(PROCESSED_FOLDER):
        file_path = os.path.join(PROCESSED_FOLDER, filename)
        if os.path.isfile(file_path):
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > cutoff_time:
                try:
                    os.remove(file_path)
                    print(f"Cleaned up processed file: {filename}")
                except Exception as e:
                    print(f"Error removing {filename}: {e}")
    
    # Clean up data_cache
    uids_to_remove = []
    for uid in list(data_cache.keys()):
        if uid in file_access_times:
            if file_access_times[uid] < cutoff_time:
                uids_to_remove.append(uid)
    
    for uid in uids_to_remove:
        data_cache.pop(uid, None)
        file_access_times.pop(uid, None)
        print(f"Cleaned up cache for UID: {uid}")
    
    # Enforce max cache size
    if len(data_cache) > MAX_CACHE_SIZE:
        # Remove oldest entries
        sorted_uids = sorted(file_access_times.items(), key=lambda x: x[1])
        for uid, _ in sorted_uids[:len(data_cache) - MAX_CACHE_SIZE]:
            data_cache.pop(uid, None)
            file_access_times.pop(uid, None)
            print(f"Removed from cache (size limit): {uid}")

def cleanup_specific_files(file_list):
    """Remove specific files from the uploads folder"""
    for filename in file_list:
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Cleaned up file: {filename}")
            except Exception as e:
                print(f"Error removing {filename}: {e}")

# ==================== HELPER FUNCTIONS ====================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_heatmap(df):
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return None
            
        corr = numeric_df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
        heatmap_path = os.path.join(STATIC_DIR, 'correlation_heatmap.png')
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(heatmap_path)
        plt.close()
        return heatmap_path
    except Exception as e:
        print(f"Error generating heatmap: {e}")
        return None

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
    
    try:
        df = pd.read_excel(filepath) if ext == 'xlsx' else pd.read_csv(filepath)
        report.append(f"✔️ Loaded dataset with shape {df.shape}.")
    except Exception as e:
        report.append(f"❌ Error loading file: {e}")
        return "", report, "", "", ""

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
    if heatmap_path:
        report.append("📊 Correlation heatmap generated.")
    else:
        report.append("ℹ️ Could not generate correlation heatmap (not enough numeric columns).")

    cleaned_path = os.path.join(PROCESSED_FOLDER, 'cleaned_' + filename)
    try:
        if ext == 'xlsx':
            df.to_excel(cleaned_path, index=False)
        else:
            df.to_csv(cleaned_path, index=False)
    except Exception as e:
        report.append(f"❌ Error saving cleaned file: {e}")

    report_path = os.path.join(PROCESSED_FOLDER, 'preprocessing_log.txt')
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
    except Exception as e:
        report.append(f"❌ Error saving report: {e}")

    # Schedule cleanup of original uploaded file
    cleanup_specific_files([filename])

    preview_html = df.head().to_html(classes='table table-striped') if not df.empty else "<p>No data to display</p>"
    return preview_html, report, cleaned_path, report_path, heatmap_path

# ==================== ROUTES ====================

@app.route('/')
def index():
    cleanup_old_files()
    try:
        return render_template('index.html')
    except Exception as e:
        return f"Error loading template: {e}<br>Template directory: {TEMPLATE_DIR}<br>Files in directory: {os.listdir(TEMPLATE_DIR) if os.path.exists(TEMPLATE_DIR) else 'Directory does not exist'}"

@app.route('/eda')
def eda():
    cleanup_old_files()
    return render_template('eda.html')

@app.route('/pre')
def pre():
    cleanup_old_files()
    return render_template('pre.html')

@app.route('/aboutus')
def aboutus():
    cleanup_old_files()
    return render_template('about.html')

@app.route('/dash')
def dash():
    cleanup_old_files()
    return render_template('dashboardind.html')

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
        
        try:
            df = pd.read_excel(filepath) if ext == 'xlsx' else pd.read_csv(filepath)
            categorical_columns = df.select_dtypes(include='object').columns.tolist()
            session['filename'] = file.filename
            column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
            return render_template('encoding_options.html', 
                                 categorical_columns=categorical_columns, 
                                 column_types=column_types, 
                                 filename=file.filename)
        except Exception as e:
            flash(f"Error reading file: {e}")
            return redirect(url_for('pre'))

    flash("Invalid file format. Please upload CSV or Excel files only.")
    return redirect(url_for('pre'))

@app.route('/preprocess', methods=['POST'])
def preprocess():
    filename = session.get('filename')
    if not filename:
        flash("Session expired or no file uploaded.")
        return redirect(url_for('pre'))

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        flash("Uploaded file not found.")
        return redirect(url_for('pre'))

    ext = filename.rsplit('.', 1)[1].lower()
    try:
        df = pd.read_excel(filepath) if ext == 'xlsx' else pd.read_csv(filepath)
    except Exception as e:
        flash(f"Error reading file: {e}")
        return redirect(url_for('pre'))

    categorical_columns = df.select_dtypes(include='object').columns.tolist()
    strategy_dict = {}
    for col in categorical_columns:
        strategy = request.form.get(f"encoding_strategy_{col}", "ignore")
        strategy_dict[col] = strategy

    preview_html, report, cleaned_path, report_path, heatmap_path = preprocess_dataset(filepath, filename, strategy_dict)
    
    # Clean up original file after preprocessing
    cleanup_specific_files([filename])
    
    heatmap_image = url_for('static', filename='correlation_heatmap.png') if heatmap_path else None
    
    return render_template('result.html', 
                         table=preview_html, 
                         report=report,
                         cleaned_filename=os.path.basename(cleaned_path),
                         log_filename=os.path.basename(report_path),
                         heatmap_image=heatmap_image)

@app.route('/download/<filename>')
def download_file(filename):
    path = os.path.join(PROCESSED_FOLDER, filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    else:
        flash("File not found.")
        return redirect(url_for('pre'))

# Upload dataset and return EDA + Correlation plot
@app.route('/upload', methods=['POST'])
def upload():
    if 'dataset' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['dataset']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file format. Please upload CSV or Excel files."}), 400

    uid = str(uuid.uuid4())
    filename = uid + "_" + file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        df = pd.read_csv(filepath) if file.filename.endswith('.csv') else pd.read_excel(filepath)
        data_cache[uid] = df
        file_access_times[uid] = time.time()

        eda = generate_eda_summary(df)
        corr_path = generate_correlation_plot(df, uid)

        # Clean up old files after new upload
        cleanup_old_files()

        return jsonify({
            "uid": uid,
            "columns": list(df.columns),
            "preview": eda["head"],
            "eda": {
                "shape": eda["shape"],
                "dtypes": eda["dtypes"],
                "missing": eda["missing_values"],
                "describe": eda["describe"],
                "tail": eda["tail"],
                "corr_path": f"/static/plots/{os.path.basename(corr_path)}" if corr_path else None
            }
        })
    except Exception as e:
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500

# Train selected model on uploaded dataset
@app.route('/train', methods=['POST'])
def train():
    uid = request.form.get('uid')
    target = request.form.get('target')
    algorithm = request.form.get('algorithm')

    if not all([uid, target, algorithm]):
        return jsonify({"error": "Missing required parameters"}), 400

    df = data_cache.get(uid)
    if df is None:
        return jsonify({"error": "Data not found"}), 400

    try:
        X_train, X_test, y_train, y_test = preprocess_data(df, target)
        model, acc, y_pred = run_model(X_train, X_test, y_train, y_test, algorithm)
        cm_path = generate_confusion_matrix_plot(y_test, y_pred, uid)
        
        # Update access time
        file_access_times[uid] = time.time()
        
        # Clean up the uploaded file for this UID
        uploaded_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.startswith(uid)]
        cleanup_specific_files(uploaded_files)

        return jsonify({
            "accuracy": round(acc * 100, 2),
            "confusion_matrix": url_for('serve_plot', filename=os.path.basename(cm_path)) if cm_path else None
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Serve static confusion matrix or correlation plots
@app.route('/static/plots/<filename>')
def serve_plot(filename):
    return send_from_directory(PLOT_FOLDER, filename)

# Dashboard
@app.route('/uploaddash', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        original_filename = file.filename
        unique_filename = str(uuid.uuid4()) + "_" + original_filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        try:
            dashboard_data = get_dashboard_data(filepath, original_filename)
            
            # Clean up the uploaded file after processing
            cleanup_specific_files([unique_filename])
            
            # Clean up old files
            cleanup_old_files()
            
            return render_template('dsahresult.html', **dashboard_data)
        except ValueError as e:
            cleanup_specific_files([unique_filename])
            return render_template('dashboardind.html', error=str(e))
        except Exception as e:
            cleanup_specific_files([unique_filename])
            return render_template('dashboardind.html', error=f"An unexpected error occurred during file processing: {e}. Please check your file.")
    else:
        return render_template('index.html', error="Invalid file type. Please upload a CSV or XLSX file.")

@app.route('/cleanup', methods=['GET'])
def manual_cleanup():
    """Manual cleanup endpoint (can be called via cron job or manually)"""
    try:
        cleanup_old_files()
        return jsonify({
            "status": "success",
            "message": "Cleanup completed successfully",
            "cache_size": len(data_cache),
            "upload_files": len(os.listdir(UPLOAD_FOLDER)),
            "processed_files": len(os.listdir(PROCESSED_FOLDER))
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# ==================== APPLICATION STARTUP ====================

if __name__ == '__main__':
    print("Starting application...")
    print(f"Current directory: {os.getcwd()}")
    print(f"Template directory: {TEMPLATE_DIR}")
    print(f"Static directory: {STATIC_DIR}")
    
    # Initial cleanup
    cleanup_old_files()
    
    # Get port from environment variable (for Render) or use default
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
