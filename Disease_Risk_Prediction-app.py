# =============================================================================
# app.py — Flask Application Entry Point
# Machine Learning-Based Disease Risk Prediction System
# =============================================================================

import os
import json
import csv
import io
from functools import wraps
from datetime import datetime

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, send_file, jsonify, abort
)

from predict import DiseasePredictor
from utils import (
    init_db, create_user, get_user_by_username, get_user_by_id,
    verify_password, save_prediction, get_user_predictions,
    get_all_predictions, get_prediction_by_id,
    generate_pdf_report, get_admin_stats, save_contact,
    get_all_users, REPORTS_DIR
)

# ─────────────────────────── App Config ───────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR   = os.path.join(BASE_DIR, 'evaluation')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

app = Flask(__name__)
app.secret_key = 'DiseaseRiskPred_SuperSecret_Key_2024!'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB upload limit

# Initialise DB and load predictor on startup
init_db()
predictor = DiseasePredictor()
try:
    predictor.load_model()
    MODEL_READY = True
except FileNotFoundError:
    MODEL_READY = False
    print("[WARNING] Model not found. Run train_model.py first.")


# =============================================================================
# Auth Decorators
# =============================================================================

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in.', 'warning')
            return redirect(url_for('login'))
        if session.get('role') != 'admin':
            flash('Admin access required.', 'danger')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated


# =============================================================================
# Context Processor — inject common vars into every template
# =============================================================================

@app.context_processor
def inject_globals():
    user = None
    if 'user_id' in session:
        user = get_user_by_id(session['user_id'])
    return dict(current_user=user, model_ready=MODEL_READY)


# =============================================================================
# Public Routes
# =============================================================================

@app.route('/')
def index():
    stats = {}
    try:
        stats = get_admin_stats()
    except Exception:
        pass
    return render_template('index.html', stats=stats)


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name    = request.form.get('name', '').strip()
        email   = request.form.get('email', '').strip()
        subject = request.form.get('subject', '').strip()
        message = request.form.get('message', '').strip()
        if name and email and message:
            save_contact(name, email, subject, message)
            flash('Thank you! Your message has been sent.', 'success')
        else:
            flash('Please fill in all required fields.', 'danger')
        return redirect(url_for('contact'))
    return render_template('contact.html')


# =============================================================================
# Auth Routes
# =============================================================================

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email    = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm  = request.form.get('confirm_password', '')

        if not all([username, email, password]):
            flash('All fields are required.', 'danger')
        elif password != confirm:
            flash('Passwords do not match.', 'danger')
        elif len(password) < 6:
            flash('Password must be at least 6 characters.', 'danger')
        else:
            # First registered user becomes admin
            from utils import get_db
            conn = get_db(); count = conn.execute('SELECT COUNT(*) FROM users').fetchone()[0]; conn.close()
            role = 'admin' if count == 0 else 'patient'
            if create_user(username, email, password, role):
                flash('Account created! Please log in.', 'success')
                return redirect(url_for('login'))
            else:
                flash('Username or email already taken.', 'danger')
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user     = get_user_by_username(username)
        if user and verify_password(password, user['password_hash'], user['salt']):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role']     = user['role']
            flash(f"Welcome back, {user['username']}! 👋", 'success')
            return redirect(url_for('admin_dashboard') if user['role'] == 'admin' else url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'danger')
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))


# =============================================================================
# Prediction Routes
# =============================================================================

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if not MODEL_READY:
        flash('Model not trained yet. Please run train_model.py first.', 'warning')
        return redirect(url_for('index'))

    if request.method == 'POST':
        try:
            form_data = {
                'Age':           int(request.form['age']),
                'Gender':        request.form['gender'],
                'BMI':           float(request.form['bmi']),
                'BloodPressure': int(request.form['blood_pressure']),
                'Glucose':       int(request.form['glucose']),
                'Insulin':       float(request.form['insulin']),
                'HeartRate':     int(request.form['heart_rate']),
                'Cholesterol':   int(request.form['cholesterol']),
                'SmokingStatus': request.form['smoking'],
                'FamilyHistory': request.form['family_history'],
                'ExerciseLevel': request.form['exercise'],
            }
        except (KeyError, ValueError) as e:
            flash(f'Invalid form input: {e}', 'danger')
            return redirect(url_for('predict'))

        result = predictor.predict(form_data)

        # Persist prediction
        user_id = session.get('user_id')
        pred_id = save_prediction(user_id, form_data, result)
        result['pred_id'] = pred_id

        return render_template('result.html', form_data=form_data, result=result)

    return render_template('predict.html')


# =============================================================================
# Model Comparison
# =============================================================================

@app.route('/comparison')
def comparison():
    results_path = os.path.join(MODELS_DIR, 'results.json')
    if not os.path.exists(results_path):
        flash('Model results not found. Please run train_model.py first.', 'warning')
        return redirect(url_for('index'))

    with open(results_path) as f:
        data = json.load(f)

    charts = [
        ('model_comparison.png',    'Model Performance Comparison'),
        ('roc_curves.png',          'ROC Curves'),
        ('confusion_matrices.png',  'Confusion Matrices'),
        ('feature_importance.png',  'Feature Importance'),
        ('correlation_heatmap.png', 'Correlation Heatmap'),
    ]
    available_charts = [
        (fn, title) for fn, title in charts
        if os.path.exists(os.path.join(EVAL_DIR, fn))
    ]

    return render_template(
        'comparison.html',
        results=data['results'],
        best_model=data['best_model'],
        charts=available_charts
    )


# Static chart serving
@app.route('/evaluation/<filename>')
def evaluation_chart(filename):
    path = os.path.join(EVAL_DIR, filename)
    if not os.path.exists(path):
        abort(404)
    return send_file(path, mimetype='image/png')


# =============================================================================
# Patient Dashboard
# =============================================================================

@app.route('/dashboard')
@login_required
def dashboard():
    predictions = get_user_predictions(session['user_id'])
    return render_template('dashboard.html', predictions=predictions)


# =============================================================================
# Admin Dashboard
# =============================================================================

@app.route('/admin')
@admin_required
def admin_dashboard():
    stats       = get_admin_stats()
    users       = get_all_users()
    predictions = get_all_predictions()
    return render_template('admin.html', stats=stats, users=users, predictions=predictions)


# =============================================================================
# PDF Report Download
# =============================================================================

@app.route('/download_report/<int:pred_id>')
@login_required
def download_report(pred_id):
    row = get_prediction_by_id(pred_id)
    if not row:
        flash('Prediction not found.', 'danger')
        return redirect(url_for('dashboard'))

    # Only the owner or admin can download
    if row['user_id'] != session.get('user_id') and session.get('role') != 'admin':
        abort(403)

    patient_data = {
        'Age': row['age'], 'Gender': row['gender'], 'BMI': row['bmi'],
        'BloodPressure': row['blood_pressure'], 'Glucose': row['glucose'],
        'Insulin': row['insulin'], 'HeartRate': row['heart_rate'],
        'Cholesterol': row['cholesterol'], 'SmokingStatus': row['smoking'],
        'FamilyHistory': row['family_history'], 'ExerciseLevel': row['exercise'],
    }

    from predict import DiseasePredictor
    dp  = DiseasePredictor()
    recs = dp.RECOMMENDATIONS.get(row['risk_level'], {})
    result = {
        'prediction':     row['prediction'],
        'probability':    row['probability'],
        'risk_level':     row['risk_level'],
        'model_name':     row['model_name'],
        'recommendation': recs,
    }

    pdf_path = generate_pdf_report(pred_id, patient_data, result)
    return send_file(pdf_path, as_attachment=True,
                     download_name=f'disease_risk_report_{pred_id}.pdf',
                     mimetype='application/pdf')


# =============================================================================
# Batch CSV Prediction
# =============================================================================

@app.route('/batch_predict', methods=['GET', 'POST'])
@login_required
def batch_predict():
    if not MODEL_READY:
        flash('Model not ready.', 'warning')
        return redirect(url_for('index'))

    results_list = []
    if request.method == 'POST':
        file = request.files.get('csv_file')
        if not file or not file.filename.endswith('.csv'):
            flash('Please upload a valid CSV file.', 'danger')
            return redirect(url_for('batch_predict'))

        try:
            stream  = io.StringIO(file.stream.read().decode('utf-8'))
            reader  = csv.DictReader(stream)
            records = list(reader)

            for i, row in enumerate(records, 1):
                try:
                    patient = {
                        'Age':           int(float(row.get('Age', 0))),
                        'Gender':        row.get('Gender', 'Male'),
                        'BMI':           float(row.get('BMI', 25)),
                        'BloodPressure': int(float(row.get('BloodPressure', 80))),
                        'Glucose':       int(float(row.get('Glucose', 100))),
                        'Insulin':       float(row.get('Insulin', 80)),
                        'HeartRate':     int(float(row.get('HeartRate', 72))),
                        'Cholesterol':   int(float(row.get('Cholesterol', 180))),
                        'SmokingStatus': row.get('SmokingStatus', 'Never'),
                        'FamilyHistory': row.get('FamilyHistory', 'No'),
                        'ExerciseLevel': row.get('ExerciseLevel', 'Moderate'),
                    }
                    res = predictor.predict(patient)
                    results_list.append({'row': i, 'patient': patient, 'result': res, 'error': None})
                except Exception as e:
                    results_list.append({'row': i, 'patient': row, 'result': None, 'error': str(e)})

            flash(f'Batch prediction complete: {len(results_list)} records processed.', 'success')
        except Exception as e:
            flash(f'Error processing CSV: {e}', 'danger')

    return render_template('batch_predict.html', results=results_list)


# =============================================================================
# API Endpoints
# =============================================================================

@app.route('/api/history')
@login_required
def api_history():
    """Return the current user's prediction history as JSON."""
    rows = get_user_predictions(session['user_id'])
    data = [dict(r) for r in rows]
    return jsonify(data)


@app.route('/api/stats')
@admin_required
def api_stats():
    return jsonify(get_admin_stats())


# =============================================================================
# Error Handlers
# =============================================================================

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(403)
def forbidden(e):
    return render_template('403.html'), 403

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  ML Disease Risk Prediction System")
    print("  http://127.0.0.1:5000")
    print("=" * 50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
