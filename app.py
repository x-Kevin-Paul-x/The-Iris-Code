from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import json # Added for serializing feature_stats
import pandas as pd
import os
import logging
import traceback # For detailed error logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("app.log"), 
                        logging.StreamHandler()      
                    ])
logger = logging.getLogger(__name__) 

app = Flask(__name__)

# --- Configuration ---
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'iris_classifier_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'iris_scaler.pkl')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'iris_label_encoder.pkl')
IRIS_DATA_PATH = os.path.join('data', 'iris.csv')
FEATURE_NAMES = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# --- Helper Functions ---
def load_resources():
    """Loads the model, scaler, label encoder, iris data, and calculates feature stats."""
    loaded_model = None
    loaded_scaler = None
    loaded_label_encoder = None
    loaded_iris_df = None
    loaded_feature_stats = {}

    if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH]):
        logger.error("Critical: Model, scaler, or label encoder file not found. Ensure training notebook was run.")
        return loaded_model, loaded_scaler, loaded_label_encoder, loaded_iris_df, loaded_feature_stats

    try:
        loaded_model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded from {MODEL_PATH}")
        loaded_scaler = joblib.load(SCALER_PATH)
        logger.info(f"Scaler loaded from {SCALER_PATH}")
        loaded_label_encoder = joblib.load(LABEL_ENCODER_PATH)
        logger.info(f"Label encoder loaded from {LABEL_ENCODER_PATH}")

        if os.path.exists(IRIS_DATA_PATH):
            loaded_iris_df = pd.read_csv(IRIS_DATA_PATH)
            logger.info(f"Original iris.csv loaded from {IRIS_DATA_PATH}.")
            for feature_name in FEATURE_NAMES:
                if feature_name in loaded_iris_df.columns:
                    loaded_feature_stats[feature_name] = {
                        'mean': loaded_iris_df[feature_name].mean(),
                        'std': loaded_iris_df[feature_name].std(),
                        'min': loaded_iris_df[feature_name].min(),
                        'max': loaded_iris_df[feature_name].max()
                    }
            logger.info(f"Feature statistics calculated: {loaded_feature_stats}")
        else:
            logger.warning(f"Original iris.csv not found at {IRIS_DATA_PATH}. Visualization/outlier detection disabled.")
    
    except FileNotFoundError as fnf_e:
        logger.error(f"A required file was not found: {fnf_e}\n{traceback.format_exc()}")
    except Exception as e:
        logger.error(f"Fatal error loading resources: {e}\n{traceback.format_exc()}")
        loaded_model, loaded_scaler, loaded_label_encoder, loaded_iris_df = None, None, None, None
        loaded_feature_stats = {}
        
    return loaded_model, loaded_scaler, loaded_label_encoder, loaded_iris_df, loaded_feature_stats

def parse_and_validate_input(form_data, current_feature_stats):
    """Parses form data, validates input features, and checks for outliers."""
    input_features = []
    input_warnings = []
    
    for feature_name in FEATURE_NAMES:
        value_str = form_data.get(feature_name)
        if value_str is None or value_str == '':
            logger.warning(f"Missing value for feature: {feature_name} from {request.remote_addr}.")
            raise ValueError(f'Missing value for {feature_name}') 
        try:
            value_float = float(value_str)
            input_features.append(value_float)

            if feature_name in current_feature_stats and current_feature_stats[feature_name]:
                stats = current_feature_stats[feature_name]
                if value_float < 0 and stats.get('min', 0) >= 0:
                     input_warnings.append(f"Warning: Input for {feature_name} ({value_float}) is negative, unusual (min: {stats.get('min', 0):.2f}).")
                
                if 'mean' in stats and 'std' in stats:
                    lower_bound = stats['mean'] - 3 * stats['std']
                    upper_bound = stats['mean'] + 3 * stats['std']
                    if not (lower_bound <= value_float <= upper_bound):
                        warning_msg = (f"Potential outlier for {feature_name}: {value_float:.2f}. "
                                       f"Typical range (mean +/- 3 std dev) approx. [{lower_bound:.2f} - {upper_bound:.2f}]. "
                                       f"Training min/max: [{stats.get('min', 'N/A'):.2f} - {stats.get('max', 'N/A'):.2f}].")
                        input_warnings.append(warning_msg)
                        logger.info(f"Outlier for {feature_name} by {request.remote_addr}: {value_float}, bounds [{lower_bound:.2f}, {upper_bound:.2f}]")
        except ValueError as ve:
            logger.warning(f"Invalid value for {feature_name}: '{value_str}' from {request.remote_addr}. Error: {ve}")
            raise ValueError(f"Invalid value for {feature_name}: '{value_str}'. Must be a number.")
            
    return input_features, input_warnings

def get_prediction_and_insights(input_features_scaled, original_input_features):
    """Makes predictions and gathers insights like confidence and feature importances."""
    
    prediction_encoded = model.predict(input_features_scaled)
    predicted_species = label_encoder.inverse_transform(prediction_encoded)[0]
    
    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_features_scaled)
        confidence = float(np.max(probabilities))

    feature_importances_dict = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_importances_dict = dict(zip(FEATURE_NAMES, map(float, importances)))

    training_data_sample_json = None
    if iris_df is not None: # iris_df is the global dataframe loaded at startup
        sampled_df = iris_df.groupby('species').apply(
            lambda x: x.sample(n=min(len(x), 30), random_state=42)
        ).reset_index(drop=True)
        
        datasets_for_chart = []
        species_colors = {
            'setosa': 'rgba(255, 99, 132, 0.6)',
            'versicolor': 'rgba(54, 162, 235, 0.6)',
            'virginica': 'rgba(75, 192, 192, 0.6)'
        }
        for species_name, group in sampled_df.groupby('species'):
            species_data = group[FEATURE_NAMES].values.tolist()
            datasets_for_chart.append({
                'label': species_name,
                'data': species_data,
                'backgroundColor': species_colors.get(species_name, 'rgba(201, 203, 207, 0.6)'),
                'borderColor': species_colors.get(species_name, 'rgba(201, 203, 207, 1)').replace('0.6', '1')
            })
        training_data_sample_json = {'datasets': datasets_for_chart}
        
    return predicted_species, confidence, feature_importances_dict, training_data_sample_json

# --- Load Resources at Startup ---
model, scaler, label_encoder, iris_df, feature_stats = load_resources()

@app.route('/')
def home():
    """Renders the home page with the input form."""
    logger.info(f"Home page requested by {request.remote_addr}.")
    # Prepare feature_stats for JSON serialization, ensuring it's an empty dict if None
    feature_stats_for_template = feature_stats if feature_stats is not None else {}
    return render_template('index.html', 
                           feature_names=FEATURE_NAMES, 
                           feature_stats_json=json.dumps(feature_stats_for_template))

@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction requests."""
    logger.info(f"Prediction requested by {request.remote_addr} with form data: {request.form.to_dict()}")
    if not model or not scaler or not label_encoder:
        logger.error("Prediction attempt failed: Model, scaler, or label encoder not loaded.")
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500

    try:
        input_features, input_warnings = parse_and_validate_input(request.form, feature_stats)
        
        input_array = np.array(input_features).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        predicted_species, confidence, feature_importances_dict, training_data_sample_json = \
            get_prediction_and_insights(input_scaled, input_features)

        # Log successful prediction details
        confidence_log_str = f"{confidence:.4f}" if confidence is not None else "N/A"
        logger.info(f"Prediction successful for {request.remote_addr}: Input={input_features}, Predicted='{predicted_species}', Confidence={confidence_log_str}")

        return jsonify({
            'prediction': predicted_species,
            'confidence': confidence,
            'feature_importances': feature_importances_dict,
            'input_features': input_features, 
            'training_data_sample': training_data_sample_json,
            'input_warnings': input_warnings
        })
        
    except ValueError as ve: # Catch validation errors from helper
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"An error occurred during prediction for {request.remote_addr}: {e}\n{traceback.format_exc()}")
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500

if __name__ == '__main__':
    if model and scaler and label_encoder:
        logger.info("Starting Flask development server on http://0.0.0.0:5000")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        logger.critical("Flask app NOT STARTED due to missing model/scaler/encoder or loading errors.")
        if not model: logger.error("Model object is None.")
        if not scaler: logger.error("Scaler object is None.")
        if not label_encoder: logger.error("LabelEncoder object is None.")
        # Check if iris_df failed to load even if other components were expected
        if iris_df is None and all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH]):
            logger.warning("Additionally, original iris.csv for visualization/stats could not be loaded.")
