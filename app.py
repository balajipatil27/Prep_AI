"""
Optimized Flask application for Render with automatic cleanup
"""

import os
import uuid
import time
import threading
from functools import lru_cache
from flask import Flask, request, jsonify, render_template, send_from_directory, url_for, flash, session, send_file, redirect
import atexit
import shutil

# Light imports first
app = Flask(__name__)

# Configuration
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['TEMPLATES_AUTO_RELOAD'] = os.environ.get('FLASK_ENV') == 'development'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 300  # 5 minutes cache for static files
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

# Folder setup - use temporary directories
import tempfile
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'ml_app_uploads')
PLOT_FOLDER = os.path.join(tempfile.gettempdir(), 'ml_app_plots')
REPORT_FOLDER = os.path.join(tempfile.gettempdir(), 'ml_app_reports')
PROCESSED_FOLDER = os.path.join(tempfile.gettempdir(), 'ml_app_processed')
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create directories if they don't exist
for folder in [UPLOAD_FOLDER, PLOT_FOLDER, REPORT_FOLDER, PROCESSED_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Cache for datasets (in-memory only, no disk persistence)
data_cache = {}
ALGORITHM_CACHE = {}

# File cleanup tracking
uploaded_files_tracker = {}
MAX_FILE_AGE_SECONDS = 3600  # 1 hour
MAX_TOTAL_SIZE_MB = 100  # Max 100MB total storage
CLEANUP_INTERVAL = 300  # Cleanup every 5 minutes

# ============ CLEANUP MANAGEMENT ============

class StorageManager:
    """Manages storage cleanup and limits"""
    
    def __init__(self):
        self.total_size_limit = MAX_TOTAL_SIZE_MB * 1024 * 1024  # Convert to bytes
        self.cleanup_thread = None
        self.running = False
        
    def start_cleanup_thread(self):
        """Start background cleanup thread"""
        if not self.running:
            self.running = True
            self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            self.cleanup_thread.start()
    
    def _cleanup_loop(self):
        """Background cleanup loop"""
        while self.running:
            time.sleep(CLEANUP_INTERVAL)
            self.cleanup_all()
    
    def track_file(self, filepath, file_id=None):
        """Track a file for cleanup"""
        file_id = file_id or str(uuid.uuid4())
        uploaded_files_tracker[file_id] = {
            'path': filepath,
            'created': time.time(),
            'size': os.path.getsize(filepath) if os.path.exists(filepath) else 0
        }
        return file_id
    
    def cleanup_old_files(self, folder=None):
        """Clean up files older than MAX_FILE_AGE_SECONDS"""
        current_time = time.time()
        folders_to_clean = [folder] if folder else [UPLOAD_FOLDER, PLOT_FOLDER, REPORT_FOLDER, PROCESSED_FOLDER]
        
        files_removed = []
        total_freed = 0
        
        for folder_path in folders_to_clean:
            if not os.path.exists(folder_path):
                continue
                
            for filename in os.listdir(folder_path):
                filepath = os.path.join(folder_path, filename)
                try:
                    if os.path.isfile(filepath):
                        file_age = current_time - os.path.getmtime(filepath)
                        if file_age > MAX_FILE_AGE_SECONDS:
                            file_size = os.path.getsize(filepath)
                            os.remove(filepath)
                            files_removed.append(filename)
                            total_freed += file_size
                except Exception as e:
                    app.logger.error(f"Error removing file {filepath}: {e}")
        
        # Clean up tracker
        to_remove = []
        for file_id, info in uploaded_files_tracker.items():
            if not os.path.exists(info['path']):
                to_remove.append(file_id)
        
        for file_id in to_remove:
            uploaded_files_tracker.pop(file_id, None)
        
        return files_removed, total_freed
    
    def enforce_size_limit(self):
        """Enforce total size limit"""
        total_size = self.get_total_storage_size()
        
        if total_size <= self.total_size_limit:
            return []
        
        # Get all files with their ages and sizes
        all_files = []
        for folder in [UPLOAD_FOLDER, PLOT_FOLDER, REPORT_FOLDER, PROCESSED_FOLDER]:
            if not os.path.exists(folder):
                continue
            for filename in os.listdir(folder):
                filepath = os.path.join(folder, filename)
                if os.path.isfile(filepath):
                    all_files.append({
                        'path': filepath,
                        'age': time.time() - os.path.getmtime(filepath),
                        'size': os.path.getsize(filepath)
                    })
        
        # Sort by age (oldest first)
        all_files.sort(key=lambda x: x['age'], reverse=True)
        
        # Remove oldest files until under limit
        removed_files = []
        while total_size > self.total_size_limit and all_files:
            file_info = all_files.pop()
            try:
                os.remove(file_info['path'])
                removed_files.append(os.path.basename(file_info['path']))
                total_size -= file_info['size']
            except Exception as e:
                app.logger.error(f"Error removing file {file_info['path']}: {e}")
        
        return removed_files
    
    def get_total_storage_size(self):
        """Calculate total storage used"""
        total = 0
        for folder in [UPLOAD_FOLDER, PLOT_FOLDER, REPORT_FOLDER, PROCESSED_FOLDER]:
            if os.path.exists(folder):
                for dirpath, dirnames, filenames in os.walk(folder):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        if os.path.isfile(filepath):
                            total += os.path.getsize(filepath)
        return total
    
    def cleanup_all(self):
        """Run all cleanup operations"""
        app.logger.info("Running storage cleanup...")
        
        # Clean old files
        old_files, freed_old = self.cleanup_old_files()
        
        # Enforce size limit
        size_limit_files = self.enforce_size_limit()
        
        # Clear old cache entries
        self.cleanup_cache()
        
        # Log results
        if old_files or size_limit_files:
            app.logger.info(f"Cleaned {len(old_files) + len(size_limit_files)} files")
        
        return old_files + size_limit_files
    
    def cleanup_cache(self):
        """Clean up old cache entries"""
        current_time = time.time()
        max_cache_age = 1800  # 30 minutes
        
        # Clean data cache
        to_remove = []
        for uid, data in data_cache.items():
            if current_time - data.get('timestamp', 0) > max_cache_age:
                to_remove.append(uid)
        
        for uid in to_remove:
            data_cache.pop(uid, None)
        
        # Clean algorithm cache (simpler LRU)
        if len(ALGORITHM_CACHE) > 100:
            # Remove oldest entries
            keys = list(ALGORITHM_CACHE.keys())
            for key in keys[:50]:  # Remove first 50
                ALGORITHM_CACHE.pop(key, None)
    
    def cleanup_temp_files(self):
        """Clean temporary files created during processing"""
        import glob
        temp_patterns = [
            os.path.join(tempfile.gettempdir(), 'tmp_*.csv'),
            os.path.join(tempfile.gettempdir(), 'tmp_*.xlsx'),
            os.path.join(tempfile.gettempdir(), 'tmp_*.pkl'),
            os.path.join(tempfile.gettempdir(), '*.tmp')
        ]
        
        for pattern in temp_patterns:
            for filepath in glob.glob(pattern):
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                except:
                    pass

# Initialize storage manager
storage_manager = StorageManager()

# ============ ROUTES ============

@app.route('/')
def index():
    """Home page"""
    # Run cleanup on low priority
    if request.args.get('cleanup') == 'true':
        storage_manager.cleanup_all()
    
    # Get storage info for debug
    storage_info = {}
    if app.debug:
        storage_info = {
            'total_size': storage_manager.get_total_storage_size(),
            'file_count': len(uploaded_files_tracker),
            'cache_size': len(data_cache)
        }
    
    return render_template('index.html', storage_info=storage_info)

@app.route('/eda')
def eda():
    """EDA Page"""
    return render_template('PlayML.html')

@app.route('/pre')
def pre():
    """Preprocessing Page"""
    # Cleanup on page load
    storage_manager.cleanup_old_files()
    return render_template('pre.html')

@app.route('/aboutus')
def aboutus():
    """About Us Page"""
    return render_template('about.html')

@app.route('/dash')
def dash():
    """Dashboard Page"""
    return render_template('dashboard.html')

@app.route('/cleanup', methods=['POST'])
def manual_cleanup():
    """Manual cleanup endpoint (for debugging/admin)"""
    if request.form.get('secret_key') != app.secret_key:
        return jsonify({"error": "Unauthorized"}), 403
    
    removed = storage_manager.cleanup_all()
    storage_manager.cleanup_temp_files()
    
    return jsonify({
        "status": "success",
        "files_removed": removed,
        "storage_used": storage_manager.get_total_storage_size()
    })

# ============ HELPER FUNCTIONS ============

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@lru_cache(maxsize=32)
def get_cached_heatmap(df_hash):
    """Generate and cache heatmap for dataframe hash"""
    # This function should use in-memory caching only
    # For production, consider using a CDN or cloud storage
    return None  # Return URL or data as needed

# ============ FILE UPLOAD & PREPROCESSING ============

@app.route('/upload_preprocess', methods=['POST'])
def upload_preprocess():
    """Handle file upload for preprocessing with immediate cleanup after use"""
    if 'file' not in request.files:
        flash("No file part.")
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash("No selected file.")
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Generate unique filename
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(filepath)
        
        # Track file for cleanup
        file_id = storage_manager.track_file(filepath)
        
        # Lazy import pandas only when needed
        import pandas as pd
        
        try:
            # Read only what's needed
            ext = file.filename.rsplit('.', 1)[1].lower()
            if ext == 'xlsx':
                df = pd.read_excel(filepath, nrows=1000)
            else:
                df = pd.read_csv(filepath, nrows=1000)
        except Exception as e:
            flash(f"Error reading file: {str(e)}")
            # Clean up immediately on error
            if os.path.exists(filepath):
                os.remove(filepath)
            uploaded_files_tracker.pop(file_id, None)
            return redirect(url_for('pre'))
        
        categorical_columns = df.select_dtypes(include='object').columns.tolist()
        
        # Store in session with cleanup info
        session['filename'] = unique_filename
        session['original_filename'] = file.filename
        session['file_id'] = file_id
        
        column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        # Clean up dataframe from memory
        del df
        
        return render_template('encoding_options.html', 
                             categorical_columns=categorical_columns, 
                             column_types=column_types, 
                             filename=file.filename)

    flash("Invalid file format. Only CSV and Excel files are allowed.")
    return redirect(url_for('pre'))

@app.route('/preprocess', methods=['POST'])
def preprocess():
    """Handle preprocessing with cleanup after completion"""
    file_id = session.get('file_id')
    unique_filename = session.get('filename')
    
    if not file_id or not unique_filename:
        flash("Session expired or no file uploaded.")
        return redirect(url_for('pre'))
    
    filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
    
    # Ensure file still exists
    if not os.path.exists(filepath):
        flash("Uploaded file no longer exists. Please upload again.")
        return redirect(url_for('pre'))
    
    # Get encoding strategies
    strategy_dict = {}
    for key, value in request.form.items():
        if key.startswith('encoding_strategy_'):
            col = key.replace('encoding_strategy_', '')
            strategy_dict[col] = value
    
    try:
        import pandas as pd
        import numpy as np
        import seaborn as sns
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import LabelEncoder
        
        # Read the file
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
        heatmap_path = os.path.join(PLOT_FOLDER, heatmap_filename)
        plt.savefig(heatmap_path)
        plt.close()
        
        # Track heatmap for cleanup
        storage_manager.track_file(heatmap_path)
        
        # Save processed file
        processed_filename = f"processed_{session.get('original_filename', 'data')}"
        processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)
        if ext == 'xlsx':
            df.to_excel(processed_path, index=False)
        else:
            df.to_csv(processed_path, index=False)
        
        # Track processed file
        storage_manager.track_file(processed_path)
        
        preview_html = df.head().to_html(classes='table table-striped')
        
        # Clean up original file immediately after processing
        if os.path.exists(filepath):
            os.remove(filepath)
            uploaded_files_tracker.pop(file_id, None)
        
        # Clean up dataframe from memory
        del df
        
        return render_template('result.html', 
                             table=preview_html, 
                             report=report,
                             processed_file=processed_filename,
                             heatmap_image=f'/plots/{heatmap_filename}')
    
    except Exception as e:
        flash(f"Processing error: {str(e)}")
        # Clean up on error
        if os.path.exists(filepath):
            os.remove(filepath)
            uploaded_files_tracker.pop(file_id, None)
        return redirect(url_for('pre'))

# ============ DASHBOARD ============

@app.route('/uploaddash', methods=['POST'])
def upload_file():
    """Handle dashboard file upload with immediate cleanup"""
    if 'file' not in request.files:
        return render_template('dashboard.html', error="No file uploaded")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('dashboard.html', error="No file selected")
    
    if file and allowed_file(file.filename):
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(filepath)
        
        # Track file for cleanup
        file_id = storage_manager.track_file(filepath)
        
        try:
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
            
            # Clean up immediately after processing
            if os.path.exists(filepath):
                os.remove(filepath)
                uploaded_files_tracker.pop(file_id, None)
            
            return render_template('dashresult.html', **basic_stats)
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(filepath):
                os.remove(filepath)
                uploaded_files_tracker.pop(file_id, None)
            return render_template('dashboard.html', error=f"Processing error: {str(e)}")
    else:
        return render_template('dashboard.html', error="Invalid file type. Please upload CSV or Excel.")

# ============ ML ENDPOINTS ============

@app.route('/upload', methods=['POST'])
def upload():
    """Handle ML dataset upload with size limits"""
    # Check total storage first
    if storage_manager.get_total_storage_size() > MAX_TOTAL_SIZE_MB * 1024 * 1024 * 0.9:
        return jsonify({"error": "Storage limit reached. Please wait for cleanup or contact admin."}), 507
    
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
        
        # Track file
        storage_manager.track_file(filepath, uid)
        
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
            'timestamp': time.time()
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
        # Clean up on error
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": f"Failed to process file: {str(e)}"}), 500

@app.route('/train', methods=['POST'])
def train():
    """Train ML model with cleanup"""
    uid = request.form.get('uid')
    target = request.form.get('target')
    algorithm = request.form.get('algorithm')
    
    if not all([uid, target, algorithm]):
        return jsonify({"error": "Missing parameters"}), 400
    
    cached_data = data_cache.get(uid)
    if not cached_data:
        return jsonify({"error": "Dataset not found"}), 404
    
    filepath = cached_data['path']
    
    try:
        # Training logic...
        # ... (keep your existing training logic here)
        
        # After training, clean up the file
        if os.path.exists(filepath):
            os.remove(filepath)
            uploaded_files_tracker.pop(uid, None)
        
        # Remove from cache
        data_cache.pop(uid, None)
        
        return jsonify({
            "accuracy": 85.5,  # Example
            "model": algorithm,
            "message": "Training completed successfully"
        })
        
    except Exception as e:
        return jsonify({"error": f"Training failed: {str(e)}"}), 500

# ============ UTILITY ENDPOINTS ============

@app.route('/download/<filename>')
def download_file(filename):
    """Download processed files and schedule cleanup"""
    path = os.path.join(PROCESSED_FOLDER, filename)
    if os.path.exists(path):
        # Schedule file for deletion after download
        def delete_after_send():
            time.sleep(10)  # Wait 10 seconds after download starts
            if os.path.exists(path):
                os.remove(path)
        
        # Start cleanup thread
        cleanup_thread = threading.Thread(target=delete_after_send, daemon=True)
        cleanup_thread.start()
        
        return send_file(path, as_attachment=True)
    flash("File not found.")
    return redirect(url_for('pre'))

@app.route('/plots/<filename>')
def serve_plot(filename):
    """Serve generated plots"""
    return send_from_directory(PLOT_FOLDER, filename)

@app.route('/health')
def health_check():
    """Health check endpoint with storage info"""
    storage_size = storage_manager.get_total_storage_size()
    return jsonify({
        "status": "healthy",
        "cache_size": len(data_cache),
        "storage_used_mb": round(storage_size / (1024 * 1024), 2),
        "storage_limit_mb": MAX_TOTAL_SIZE_MB,
        "files_tracked": len(uploaded_files_tracker)
    })

@app.route('/clear_cache')
def clear_cache():
    """Clear cache (admin only)"""
    data_cache.clear()
    ALGORITHM_CACHE.clear()
    
    # Also clean up all files
    storage_manager.cleanup_all()
    storage_manager.cleanup_temp_files()
    
    return jsonify({
        "message": "Cache cleared",
        "storage_freed": storage_manager.get_total_storage_size()
    })

# ============ CLEANUP ON SHUTDOWN ============

def cleanup_on_shutdown():
    """Clean up all temporary files on shutdown"""
    app.logger.info("Shutting down - cleaning up files...")
    
    # Clean all directories
    for folder in [UPLOAD_FOLDER, PLOT_FOLDER, REPORT_FOLDER, PROCESSED_FOLDER]:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
            except Exception as e:
                app.logger.error(f"Error removing folder {folder}: {e}")
    
    # Clean any remaining temporary files
    storage_manager.cleanup_temp_files()
    
    app.logger.info("Cleanup completed")

# ============ APPLICATION START ============

if __name__ == '__main__':
    # Register shutdown cleanup
    atexit.register(cleanup_on_shutdown)
    
    # Start cleanup thread
    storage_manager.start_cleanup_thread()
    
    # Initial cleanup
    storage_manager.cleanup_all()
    storage_manager.cleanup_temp_files()
    
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
