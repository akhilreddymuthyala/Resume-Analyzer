from flask import Flask, render_template, request, jsonify
import joblib
import PyPDF2
import os
from werkzeug.utils import secure_filename
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download NLTK data with error handling
def download_nltk_data():
    """Download required NLTK data"""
    nltk_packages = [
        'punkt',
        'punkt_tab',
        'stopwords',
        'averaged_perceptron_tagger'
    ]
    
    for package in nltk_packages:
        try:
            nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
            logger.info(f"✓ NLTK package '{package}' already downloaded")
        except LookupError:
            try:
                nltk.download(package, quiet=True)
                logger.info(f"✓ Downloaded NLTK package '{package}'")
            except Exception as e:
                logger.warning(f"⚠ Failed to download '{package}': {e}")

download_nltk_data()

# ============================================================================
# CRITICAL: Define ImprovedTextPreprocessor BEFORE loading any pickle files
# This must match EXACTLY the class definition used during model training
# ============================================================================

class ImprovedTextPreprocessor:
    """
    Enhanced text preprocessing for resume and job description text.
    This class must be defined before loading pickled objects that reference it.
    """
    def __init__(self):
        try:
            # Use a smaller set of stopwords to preserve technical context
            self.stop_words = set(stopwords.words('english')) - {
                'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 
                'over', 'under', 'again', 'further', 'then', 'once'
            }
        except Exception as e:
            logger.warning(f"Could not load stopwords: {e}. Using empty set.")
            self.stop_words = set()
    
    def clean_text(self, text):
        """Clean and preprocess text while preserving technical terms"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Keep alphanumeric, spaces, and important symbols like +, #, .
        text = re.sub(r'[^a-zA-Z0-9\s\+\#\.]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_filter(self, text):
        """Tokenize and remove only stopwords, NO stemming"""
        try:
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove only stopwords, keep all meaningful terms
            tokens = [token for token in tokens 
                      if token not in self.stop_words and len(token) > 1]
            
            return ' '.join(tokens)
        except Exception as e:
            logger.warning(f"Tokenization failed: {e}. Returning cleaned text.")
            return text
    
    def preprocess(self, text):
        """Complete preprocessing pipeline"""
        text = self.clean_text(text)
        text = self.tokenize_and_filter(text)
        return text

# ============================================================================
# Initialize Flask App
# ============================================================================

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ============================================================================
# Load Dataset and Model Components
# ============================================================================

# Global variables for model components
model = None
vectorizer = None
label_encoder = None
preprocessor = None
df = None

def load_dataset():
    """Load the dataset CSV file"""
    global df
    try:
        df = pd.read_csv('ats_claude.csv')
        logger.info(f"✓ Dataset loaded successfully: {len(df)} records, {len(df['role'].unique())} unique roles")
        return True
    except FileNotFoundError:
        logger.error("❌ Dataset file 'ats_claude.csv' not found")
        df = pd.DataFrame(columns=['role', 'skills', 'experience_description'])
        return False
    except Exception as e:
        logger.error(f"❌ Error loading dataset: {e}")
        df = pd.DataFrame(columns=['role', 'skills', 'experience_description'])
        return False

def load_model_components():
    """Load trained model and preprocessing components"""
    global model, vectorizer, label_encoder, preprocessor
    
    required_files = {
        'model': 'ats_model_improved.pkl',
        'vectorizer': 'tfidf_vectorizer_improved.pkl',
        'label_encoder': 'label_encoder_improved.pkl',
        'preprocessor': 'text_preprocessor_improved.pkl'
    }
    
    # Check if all files exist
    missing_files = [f for f in required_files.values() if not os.path.exists(f)]
    if missing_files:
        logger.error(f"❌ Missing model files: {', '.join(missing_files)}")
        return False
    
    try:
        model = joblib.load(required_files['model'])
        vectorizer = joblib.load(required_files['vectorizer'])
        label_encoder = joblib.load(required_files['label_encoder'])
        preprocessor = joblib.load(required_files['preprocessor'])
        
        logger.info("✓ All model components loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading model components: {e}")
        model = None
        vectorizer = None
        label_encoder = None
        preprocessor = None
        return False

# Initialize on startup
logger.info("="*80)
logger.info("🚀 INITIALIZING ATS RESUME CLASSIFIER")
logger.info("="*80)

dataset_loaded = load_dataset()
model_loaded = load_model_components()

if not dataset_loaded:
    logger.warning("⚠ Server starting with empty dataset")
if not model_loaded:
    logger.warning("⚠ Server starting without model (limited functionality)")

logger.info("="*80)

# ============================================================================
# Utility Functions
# ============================================================================

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file with error handling"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception as e:
                    logger.warning(f"Could not extract text from page {page_num + 1}: {e}")
            
            if not text.strip():
                logger.warning("PDF contains no extractable text")
                return None
                
            return text
            
    except Exception as e:
        logger.error(f"Error reading PDF file: {e}")
        return None

def extract_skills_from_text(text, role_skills):
    """Extract skills from resume text with improved matching"""
    text_lower = text.lower()
    found_skills = []
    
    # Common skill variations and abbreviations
    skill_variations = {
        'machine learning': ['ml', 'machine learning', 'machine-learning'],
        'artificial intelligence': ['ai', 'artificial intelligence'],
        'deep learning': ['dl', 'deep learning', 'deep-learning'],
        'natural language processing': ['nlp', 'natural language processing'],
        'computer vision': ['cv', 'computer vision'],
        'python': ['python', 'python3', 'python2'],
        'javascript': ['javascript', 'js', 'ecmascript'],
        'typescript': ['typescript', 'ts'],
        'c++': ['c++', 'cpp', 'cplusplus'],
        'c#': ['c#', 'csharp', 'c-sharp'],
        'react': ['react', 'reactjs', 'react.js'],
        'node': ['node', 'nodejs', 'node.js'],
        'angular': ['angular', 'angularjs', 'angular.js'],
        'vue': ['vue', 'vuejs', 'vue.js'],
    }
    
    for skill in role_skills:
        skill_lower = skill.strip().lower()
        
        # Direct match with word boundaries
        pattern = r'\b' + re.escape(skill_lower) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.append(skill.strip())
            continue
        
        # Check variations
        for standard_skill, variations in skill_variations.items():
            if skill_lower in variations or any(v == skill_lower for v in variations):
                for variant in variations:
                    pattern = r'\b' + re.escape(variant) + r'\b'
                    if re.search(pattern, text_lower):
                        found_skills.append(skill.strip())
                        break
                if skill.strip() in found_skills:
                    break
    
    return list(set(found_skills))

def calculate_ats_score(resume_text, target_role):
    """Calculate comprehensive ATS score for the resume"""
    
    # Validate inputs
    if not resume_text or not resume_text.strip():
        return {'error': 'Resume text is empty'}
    
    if df is None or df.empty:
        return {'error': 'Dataset not available'}
    
    if model is None or vectorizer is None or label_encoder is None or preprocessor is None:
        return {'error': 'Model components not loaded'}
    
    try:
        # Preprocess resume text
        processed_text = preprocessor.preprocess(resume_text)
        
        if not processed_text.strip():
            return {'error': 'Resume contains no meaningful text after preprocessing'}
        
        # Vectorize
        text_tfidf = vectorizer.transform([processed_text])
        
        # Get predictions and probabilities
        prediction_proba = model.predict_proba(text_tfidf)[0]
        predicted_role_idx = model.predict(text_tfidf)[0]
        predicted_role = label_encoder.inverse_transform([predicted_role_idx])[0]
        
        # Get probability for target role
        try:
            target_role_idx = np.where(label_encoder.classes_ == target_role)[0][0]
            target_role_probability = prediction_proba[target_role_idx]
        except IndexError:
            return {'error': f'Target role "{target_role}" not found in model'}
        
        # Get required skills for target role
        role_data = df[df['role'] == target_role]
        if role_data.empty:
            return {'error': f'No data found for role "{target_role}"'}
        
        role_data = role_data.iloc[0]
        required_skills = [skill.strip() for skill in str(role_data['skills']).split(',') if skill.strip()]
        
        # Extract skills from resume
        found_skills = extract_skills_from_text(resume_text, required_skills)
        missing_skills = [skill for skill in required_skills if skill not in found_skills]
        
        # Calculate skill match percentage
        skill_match_percentage = (len(found_skills) / len(required_skills)) * 100 if required_skills else 0
        
        # Calculate text similarity
        role_desc = f"{role_data['skills']}. {role_data['experience_description']}"
        role_processed = preprocessor.preprocess(role_desc)
        role_tfidf = vectorizer.transform([role_processed])
        similarity_score = cosine_similarity(text_tfidf, role_tfidf)[0][0]
        
        # Calculate composite ATS score (weighted average)
        # 40% model confidence + 40% skill match + 20% text similarity
        ats_score = (
            target_role_probability * 0.40 +
            (skill_match_percentage / 100) * 0.40 +
            similarity_score * 0.20
        ) * 100
        
        # Get top 3 predicted roles
        top_3_indices = np.argsort(prediction_proba)[-3:][::-1]
        top_3_roles = [
            {
                'role': label_encoder.inverse_transform([idx])[0],
                'probability': round(float(prediction_proba[idx]) * 100, 2)
            }
            for idx in top_3_indices
        ]
        
        return {
            'success': True,
            'target_role': target_role,
            'predicted_role': predicted_role,
            'ats_score': round(float(ats_score), 2),
            'model_confidence': round(float(target_role_probability) * 100, 2),
            'skill_match_percentage': round(float(skill_match_percentage), 2),
            'text_similarity': round(float(similarity_score) * 100, 2),
            'required_skills': required_skills,
            'found_skills': found_skills,
            'missing_skills': missing_skills,
            'total_required': len(required_skills),
            'total_found': len(found_skills),
            'total_missing': len(missing_skills),
            'top_3_predictions': top_3_roles,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error calculating ATS score: {e}", exc_info=True)
        return {'error': f'Error analyzing resume: {str(e)}'}

# ============================================================================
# Flask Routes
# ============================================================================

@app.route('/')
def index():
    """Render the home page"""
    if df is None or df.empty:
        return render_template('error.html', 
                             error_message="Dataset not available. Please contact administrator."), 500
    
    available_roles = sorted(df['role'].unique().tolist())
    
    return render_template('index.html', 
                         roles=available_roles,
                         model_loaded=model is not None)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze the uploaded resume"""
    
    # Check if model is loaded
    if model is None:
        return jsonify({'error': 'Model not loaded. Service temporarily unavailable.'}), 503
    
    if df is None or df.empty:
        return jsonify({'error': 'Dataset not available. Service temporarily unavailable.'}), 503
    
    # Validate file upload
    if 'resume' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['resume']
    target_role = request.form.get('role')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not target_role:
        return jsonify({'error': 'No role selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only PDF files are allowed.'}), 400
    
    # Process the file
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_filename = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    
    try:
        # Save the uploaded file
        file.save(filepath)
        logger.info(f"Processing resume: {unique_filename} for role: {target_role}")
        
        # Extract text from PDF
        resume_text = extract_text_from_pdf(filepath)
        
        if not resume_text:
            return jsonify({'error': 'Could not extract text from PDF. Please ensure the PDF contains readable text.'}), 400
        
        # Calculate ATS score
        result = calculate_ats_score(resume_text, target_role)
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 400
        
        logger.info(f"Analysis complete: ATS Score = {result['ats_score']}")
        
        return render_template('result.html', result=result)
        
    except Exception as e:
        logger.error(f"Error processing resume: {e}", exc_info=True)
        return jsonify({'error': f'Error processing resume: {str(e)}'}), 500
        
    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.info(f"Cleaned up temporary file: {unique_filename}")
            except Exception as e:
                logger.warning(f"Could not delete temporary file: {e}")

@app.route('/api/roles', methods=['GET'])
def get_roles():
    """API endpoint to get available roles"""
    if df is None or df.empty:
        return jsonify({'error': 'Dataset not available'}), 503
    
    try:
        available_roles = sorted(df['role'].unique().tolist())
        return jsonify({
            'success': True,
            'roles': available_roles,
            'count': len(available_roles)
        })
    except Exception as e:
        logger.error(f"Error fetching roles: {e}")
        return jsonify({'error': 'Could not fetch roles'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'dataset_loaded': df is not None and not df.empty,
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {e}", exc_info=True)
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(413)
def file_too_large(e):
    """Handle file size errors"""
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

# ============================================================================
# Application Entry Point
# ============================================================================

if __name__ == '__main__':
    # Validate startup conditions
    if not dataset_loaded:
        logger.error("❌ Cannot start server - dataset failed to load")
        exit(1)
    
    if not model_loaded:
        logger.warning("⚠ Starting server without model - functionality will be limited")
    
    # Print startup information
    print("\n" + "="*80)
    print("🚀 ATS RESUME CLASSIFIER SERVER STARTING")
    print("="*80)
    print(f"📍 Server URL: http://127.0.0.1:5000")
    print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}/")
    print(f"📊 Model loaded: {'✓' if model_loaded else '✗'}")
    print(f"📈 Dataset loaded: {'✓' if dataset_loaded else '✗'}")
    
    if dataset_loaded and df is not None:
        print(f"🎯 Available roles: {len(df['role'].unique())}")
    
    print(f"🔧 Debug mode: {'ON' if os.environ.get('FLASK_ENV') == 'development' else 'OFF'}")
    print("="*80 + "\n")
    
    # Start server
    # For production, use a WSGI server like Gunicorn instead of Flask's built-in server
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    app.run(
        debug=debug_mode,
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000))
    )