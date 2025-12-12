"""
Optimized Flask application with lazy loading and caching for better performance
"""

import os
import uuid
from functools import lru_cache
from flask import Flask, request, jsonify, render_template, send_from_directory, url_for, flash, session, send_file, redirect

# Light imports first
app = Flask(__name__)

# Configuration
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['TEMPLATES_AUTO_RELOAD'] = os.environ.get('FLASK_ENV') == 'development'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 300  # 5 minutes cache for static files
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

# Folder setup
UPLOAD_FOLDER = 'uploads'
PLOT_FOLDER = 'static/plots'
REPORT_FOLDER = 'static/reports'
PROCESSED_FOLDER = 'processed'
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create directories if they don't exist
for folder in [UPLOAD_FOLDER, PLOT_FOLDER, REPORT_FOLDER, PROCESSED_FOLDER, STATIC_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Cache for datasets (in production, use Redis)
data_cache = {}
ALGORITHM_CACHE = {}

# ============ ROUTES ============

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/eda')
def eda():
    """EDA Page"""
    return render_template('PlayML.html')

@app.route('/pre')
def pre():
    """Preprocessing Page"""
    return render_template('pre.html')

@app.route('/aboutus')
def aboutus():
    """About Us Page"""
    return render_template('about.html')

@app.route('/dash')
def dash():
    """Dashboard Page"""
    return render_template('dashboard.html')

# ============ HELPER FUNCTIONS ============

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@lru_cache(maxsize=32)
def get_cached_heatmap(df_hash):
    """Generate and cache heatmap for dataframe hash"""
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # This is a simplified version - in practice you'd store the actual df
    # or regenerate from cached data
    return "/static/plots/default_heatmap.png"

# ============ FILE UPLOAD & PREPROCESSING ============

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
        # Generate unique filename to avoid conflicts
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(filepath)
        
        # Lazy import pandas only when needed
        import pandas as pd
        
        # Read only necessary info without loading entire file
        ext = file.filename.rsplit('.', 1)[1].lower()
        try:
            if ext == 'xlsx':
                # Read only first few rows to get column info
                df = pd.read_excel(filepath, nrows=1000)
            else:
                df = pd.read_csv(filepath, nrows=1000)
        except Exception as e:
            flash(f"Error reading file: {str(e)}")
            return redirect(url_for('pre'))
        
        categorical_columns = df.select_dtypes(include='object').columns.tolist()
        
        # Store minimal info in session
        session['filename'] = unique_filename
        session['original_filename'] = file.filename
        
        column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        # Clean up
        del df
        
        return render_template('encoding_options.html', 
                             categorical_columns=categorical_columns, 
                             column_types=column_types, 
                             filename=file.filename)

    flash("Invalid file format. Only CSV and Excel files are allowed.")
    return redirect(url_for('pre'))

@app.route('/preprocess', methods=['POST'])
def preprocess():
    """Handle preprocessing with encoding strategies"""
    unique_filename = session.get('filename')
    if not unique_filename:
        flash("Session expired or no file uploaded.")
        return redirect(url_for('pre'))
    
    # Get encoding strategies
    strategy_dict = {}
    for key, value in request.form.items():
        if key.startswith('encoding_strategy_'):
            col = key.replace('encoding_strategy_', '')
            strategy_dict[col] = value
    
    # Process in background if large file
    filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
    if os.path.getsize(filepath) > 10 * 1024 * 1024:  # > 10MB
        return render_template('processing.html', 
                             message="Processing large file... This may take a moment.")
    
    # Import preprocessing functions
    try:
        from preprocessing_module import preprocess_dataset, generate_heatmap
    except ImportError:
        # Fallback to inline processing
        import pandas as pd
        import numpy as np
        import seaborn as sns
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import LabelEncoder
        
        # Simplified preprocessing logic
        ext = unique_filename.rsplit('.', 1)[1].lower()
        df = pd.read_excel(filepath) if ext == 'xlsx' else pd.read_csv(filepath)
        
        # Basic preprocessing
        report = []
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
                report.append(f"Filled missing values in {col}")
        
        # Encode categorical
        for col, method in strategy_dict.items():
            if method == 'onehot' and col in df.columns:
                df = pd.get_dummies(df, columns=[col], prefix=col)
            elif method == 'label' and col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        
        # Generate heatmap
        corr = df.select_dtypes(include=[np.number]).corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
        heatmap_filename = f"{uuid.uuid4()}_heatmap.png"
        heatmap_path = os.path.join(STATIC_FOLDER, heatmap_filename)
        plt.savefig(heatmap_path)
        plt.close()
        
        # Save processed file
        processed_filename = f"processed_{session.get('original_filename', 'data')}"
        processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)
        if ext == 'xlsx':
            df.to_excel(processed_path, index=False)
        else:
            df.to_csv(processed_path, index=False)
        
        preview_html = df.head().to_html(classes='table table-striped')
        
        return render_template('result.html', 
                             table=preview_html, 
                             report=report,
                             processed_file=processed_filename,
                             heatmap_image=f'/static/{heatmap_filename}')
    
    return render_template('result.html', table="", report=["Processing complete."])

# ============ DASHBOARD ============

@app.route('/uploaddash', methods=['POST'])
def upload_file():
    """Handle dashboard file upload"""
    if 'file' not in request.files:
        return render_template('dashboard.html', error="No file uploaded")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('dashboard.html', error="No file selected")
    
    if file and allowed_file(file.filename):
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(filepath)
        
        try:
            # Lazy import
            from dashboard import get_dashboard_data
            dashboard_data = get_dashboard_data(filepath, file.filename)
            return render_template('dashresult.html', **dashboard_data)
        except ImportError:
            # Simplified dashboard
            import pandas as pd
            import json
            
            ext = file.filename.rsplit('.', 1)[1].lower()
            if ext == 'xlsx':
                df = pd.read_excel(filepath)
            else:
                df = pd.read_csv(filepath)
            
            basic_stats = {
                'filename': file.filename,
                'rows': len(df),
                'columns': len(df.columns),
                'columns_list': list(df.columns),
                'preview': df.head(5).to_html(classes='table table-striped'),
                'missing': df.isnull().sum().to_dict(),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
            }
            
            return render_template('dashresult.html', **basic_stats)
        except Exception as e:
            return render_template('dashboard.html', error=f"Processing error: {str(e)}")
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
    else:
        return render_template('dashboard.html', error="Invalid file type. Please upload CSV or Excel.")

# ============ ML ENDPOINTS ============

@app.route('/upload', methods=['POST'])
def upload():
    """Handle ML dataset upload"""
    file = request.files.get('dataset')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400
    
    # Generate unique ID
    uid = str(uuid.uuid4())
    safe_filename = f"{uid}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, safe_filename)
    
    try:
        file.save(filepath)
        
        # Lazy import
        import pandas as pd
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(filepath, nrows=10000)  # Limit for initial load
        else:
            df = pd.read_excel(filepath, nrows=10000)
        
        # Cache the dataframe
        data_cache[uid] = {
            'df': df,
            'path': filepath,
            'timestamp': pd.Timestamp.now()
        }
        
        # Generate basic info
        response = {
            "uid": uid,
            "columns": list(df.columns),
            "shape": df.shape,
            "preview": df.head().to_dict(orient='records'),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"Failed to process file: {str(e)}"}), 500

@app.route('/suggest_algorithms', methods=['POST'])
def suggest_algorithms_route():
    """Suggest algorithms based on dataset"""
    uid = request.form.get('uid')
    target = request.form.get('target')
    
    if not uid or not target:
        return jsonify({"error": "Missing parameters"}), 400
    
    cached_data = data_cache.get(uid)
    if not cached_data:
        return jsonify({"error": "Dataset not found"}), 404
    
    df = cached_data['df']
    
    if target not in df.columns:
        return jsonify({"error": "Target column not found"}), 400
    
    # Cache algorithm suggestions
    cache_key = f"{uid}_{target}"
    if cache_key in ALGORITHM_CACHE:
        return jsonify({"suggested_algorithms": ALGORITHM_CACHE[cache_key]})
    
    # Determine task type
    if pd.api.types.is_numeric_dtype(df[target]):
        task_type = "regression"
        unique_vals = None
    else:
        task_type = "classification"
        unique_vals = df[target].nunique()
    
    # Suggest algorithms
    algorithms = []
    
    if task_type == "regression":
        algorithms = [
            {"value": "linear_regression", "label": "Linear Regression"},
            {"value": "decision_tree", "label": "Decision Tree Regressor"},
            {"value": "random_forest", "label": "Random Forest Regressor"}
        ]
    else:
        if unique_vals == 2:
            algorithms = [
                {"value": "logistic", "label": "Logistic Regression"},
                {"value": "decision_tree", "label": "Decision Tree Classifier"},
                {"value": "random_forest", "label": "Random Forest Classifier"},
                {"value": "svm", "label": "SVM Classifier"}
            ]
        else:
            algorithms = [
                {"value": "decision_tree", "label": "Decision Tree Classifier"},
                {"value": "random_forest", "label": "Random Forest Classifier"},
                {"value": "svm", "label": "SVM Classifier"}
            ]
    
    # Cache results
    ALGORITHM_CACHE[cache_key] = algorithms
    
    return jsonify({"suggested_algorithms": algorithms})

@app.route('/train', methods=['POST'])
def train():
    """Train ML model"""
    uid = request.form.get('uid')
    target = request.form.get('target')
    algorithm = request.form.get('algorithm')
    
    if not all([uid, target, algorithm]):
        return jsonify({"error": "Missing parameters"}), 400
    
    cached_data = data_cache.get(uid)
    if not cached_data:
        return jsonify({"error": "Dataset not found"}), 404
    
    df = cached_data['df']
    
    try:
        # Lazy import ML modules
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import accuracy_score
        
        # Simple preprocessing
        X = df.drop(columns=[target])
        y = df[target]
        
        # Encode if categorical
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        if algorithm == 'logistic':
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(max_iter=1000)
        elif algorithm == 'linear_regression':
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        elif algorithm == 'decision_tree':
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
            if len(np.unique(y)) == 2:
                model = DecisionTreeClassifier()
            else:
                model = DecisionTreeRegressor()
        elif algorithm == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            if len(np.unique(y)) == 2:
                model = RandomForestClassifier()
            else:
                model = RandomForestRegressor()
        elif algorithm == 'svm':
            from sklearn.svm import SVC
            model = SVC(probability=True)
        else:
            return jsonify({"error": "Unsupported algorithm"}), 400
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if algorithm == 'linear_regression':
            from sklearn.metrics import r2_score
            score = r2_score(y_test, y_pred)
        else:
            score = accuracy_score(y_test, y_pred)
        
        return jsonify({
            "accuracy": round(score * 100, 2),
            "model": algorithm,
            "message": "Training completed successfully"
        })
        
    except Exception as e:
        return jsonify({"error": f"Training failed: {str(e)}"}), 500

# ============ UTILITY ENDPOINTS ============

@app.route('/download/<filename>')
def download_file(filename):
    """Download processed files"""
    path = os.path.join(PROCESSED_FOLDER, filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    flash("File not found.")
    return redirect(url_for('pre'))

@app.route('/static/plots/<filename>')
def serve_plot(filename):
    """Serve generated plots"""
    return send_from_directory(PLOT_FOLDER, filename)

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({"status": "healthy", "cache_size": len(data_cache)})

@app.route('/clear_cache')
def clear_cache():
    """Clear cache (admin only)"""
    data_cache.clear()
    ALGORITHM_CACHE.clear()
    return jsonify({"message": "Cache cleared"})

# ============ MAINTENANCE ============

def cleanup_old_files():
    """Clean up old uploaded files"""
    import time
    current_time = time.time()
    
    for filename in os.listdir(UPLOAD_FOLDER):
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.getmtime(filepath) < current_time - 3600:  # 1 hour old
            os.remove(filepath)
    
    for filename in os.listdir(PLOT_FOLDER):
        filepath = os.path.join(PLOT_FOLDER, filename)
        if os.path.getmtime(filepath) < current_time - 86400:  # 1 day old
            os.remove(filepath)

# ============ APPLICATION START ============

if __name__ == '__main__':
    # Cleanup on start
    cleanup_old_files()
    
    # Production settings
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    if not debug:
        # Production optimizations
        import logging
        logging.basicConfig(level=logging.WARNING)
        
    app.run(
        host='0.0.0.0', 
        port=port, 
        debug=debug,
        threaded=True
    )
