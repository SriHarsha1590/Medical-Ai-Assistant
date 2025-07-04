import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

import logging
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, session, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import google.cloud
import fitz  # PyMuPDF
from transformers import pipeline
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import os
import requests
import mysql.connector
from mysql.connector import Error
import openai  # Import OpenAI for future AI-powered queries
import re
from google.cloud import vision
import io
from difflib import SequenceMatcher

# Set up logging
logging.basicConfig(
    level=logging.WARNING,  # Changed from INFO/DEBUG to WARNING to reduce terminal output
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler()]
)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'your_secret_key'

# --- Database Setup (MySQL) ---
def get_db_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='1234',
        database='medical_ai'
    )

def create_users_table():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('CREATE DATABASE IF NOT EXISTS medical_ai')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(255) UNIQUE NOT NULL,
            password VARCHAR(255) NOT NULL
        )
    ''')
    conn.commit()
    cursor.close()
    conn.close()

create_users_table()

users = {}

summarizer = pipeline('summarization', model='sshleifer/distilbart-cnn-12-6')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Registration Route ---
@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'username' in session:
        flash('You are already logged in.', 'info')
        return redirect(url_for('upload_file'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
        user = cursor.fetchone()
        if user:
            cursor.close()
            conn.close()
            flash('Username already exists.', 'danger')
            return render_template('register.html')
        hashed_pw = generate_password_hash(password)
        cursor.execute('INSERT INTO users (username, password) VALUES (%s, %s)', (username, hashed_pw))
        conn.commit()
        cursor.close()
        conn.close()
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

# --- Login Route ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        if user and check_password_hash(user['password'], password):
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('upload_file'))
        else:
            flash('Invalid username or password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('Logged out successfully.', 'success')
    return redirect(url_for('login'))

# --- Home/Upload Route ---
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if 'username' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('medicine_summary', filename=filename))
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# --- Extract Text from File ---
def extract_text_from_file(filepath):
    ext = filepath.rsplit('.', 1)[1].lower()
    text = ''
    if ext in ['png', 'jpg', 'jpeg']:
        client = vision.ImageAnnotatorClient()
        with io.open(filepath, 'rb') as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        response = client.text_detection(image=image)
        texts = response.text_annotations
        if texts:
            text = texts[0].description
    elif ext == 'pdf':
        doc = fitz.open(filepath)
        for page in doc:
            text += page.get_text()
    return text

# --- Symptom Checker ---
def check_symptoms(symptoms):
    tips = []
    medicines = []
    descriptions = []
    symptoms = [s.strip().lower() for s in symptoms.split(',')]
    if 'fever' in symptoms:
        descriptions.append('Fever is a temporary increase in your body temperature, often due to an illness. It is a common sign of infection. High fever may indicate a serious infection and should be monitored closely.')
        tips.extend([
            'Stay hydrated and rest as much as possible.',
            'Monitor your temperature regularly.',
            'Wear light clothing and keep the room cool.',
            'Consult a doctor if fever persists for more than 3 days or is very high.',
            'Take a lukewarm bath to help reduce temperature.',
            'Avoid self-medicating with antibiotics.'
        ])
        medicines.extend([
            'Paracetamol (consult doctor for dosage)',
            'Ibuprofen (if not allergic, consult doctor)',
            'ORS (Oral Rehydration Solution) for dehydration',
            'Acetaminophen (for pain and fever, consult doctor)'
        ])
    if 'cough' in symptoms:
        descriptions.append('Coughing is a reflex that helps clear your airways of irritants and mucus. It can be caused by infections, allergies, or other conditions. Persistent cough may require medical attention.')
        tips.extend([
            'Drink warm fluids and avoid cold drinks.',
            'Use cough lozenges or honey (if not allergic).',
            'Avoid irritants like smoke and dust.',
            'See a doctor if cough lasts more than 2 weeks or is severe.',
            'Use a humidifier to keep air moist.',
            'Elevate your head while sleeping.'
        ])
        medicines.extend([
            'Cough syrup (consult pharmacist/doctor)',
            'Honey and ginger (home remedy, if not allergic)',
            'Steam inhalation for relief',
            'Dextromethorphan (cough suppressant, consult doctor)',
            'Guaifenesin (expectorant, consult doctor)'
        ])
    if 'headache' in symptoms:
        descriptions.append('A headache is pain or discomfort in the head or face area. It can be caused by stress, dehydration, or underlying illness. Severe or sudden headaches may require urgent care.')
        tips.extend([
            'Rest in a quiet, dark room.',
            'Apply a cool compress to your forehead.',
            'Stay hydrated and avoid skipping meals.',
            'Consult a doctor if headache is severe or recurrent.',
            'Practice relaxation techniques like deep breathing.',
            'Avoid excessive screen time.'
        ])
        medicines.extend([
            'Paracetamol or ibuprofen (consult doctor for dosage)',
            'Aspirin (not for children, consult doctor)',
            'Avoid self-medication for chronic headaches',
            'Sumatriptan (for migraines, prescription only)'
        ])
    if 'cold' in symptoms:
        descriptions.append('A common cold is a viral infection of your nose and throat. It usually resolves on its own. Symptoms include runny nose, sneezing, and mild fever.')
        tips.extend([
            'Rest and drink plenty of fluids.',
            'Use saline nasal drops for congestion.',
            'Gargle with warm salt water for sore throat.',
            'Wash your hands frequently to prevent spread.',
            'Avoid close contact with others.'
        ])
        medicines.extend([
            'Decongestant tablets or syrups (consult doctor)',
            'Paracetamol for fever or pain',
            'Antihistamines for runny nose (consult doctor)',
            'Vitamin C and zinc supplements (consult doctor)'
        ])
    if 'sore throat' in symptoms:
        descriptions.append('A sore throat is pain, scratchiness or irritation of the throat that often worsens when you swallow. It can be caused by viral or bacterial infections.')
        tips.extend([
            'Gargle with warm salt water.',
            'Drink warm liquids and avoid irritants.',
            'Use throat lozenges.',
            'Avoid spicy or acidic foods.',
            'Rest your voice.'
        ])
        medicines.extend([
            'Throat lozenges',
            'Mild pain relievers (consult doctor)',
            'Antibiotics (only if prescribed by doctor)',
            'Antiseptic mouthwash (consult doctor)'
        ])
    if 'vomiting' in symptoms:
        descriptions.append('Vomiting is the forceful expulsion of stomach contents through the mouth. It can be caused by infections, food poisoning, or other illnesses.')
        tips.extend([
            'Sip clear fluids to stay hydrated.',
            'Avoid solid food until vomiting stops.',
            'Rest and avoid strong odors.',
            'Seek medical help if vomiting is severe or persistent.'
        ])
        medicines.extend([
            'ORS (Oral Rehydration Solution)',
            'Domperidone (consult doctor)',
            'Ondansetron (consult doctor)'
        ])
    if 'diarrhea' in symptoms:
        descriptions.append('Diarrhea is frequent, loose, or watery bowel movements. It can lead to dehydration and may be caused by infections or food intolerance.')
        tips.extend([
            'Drink plenty of fluids, especially ORS.',
            'Avoid dairy, fatty, or spicy foods.',
            'Eat small, bland meals.',
            'Seek medical help if diarrhea is severe or lasts more than 2 days.'
        ])
        medicines.extend([
            'ORS (Oral Rehydration Solution)',
            'Loperamide (consult doctor)',
            'Zinc supplements (consult doctor)'
        ])
    if 'stomach pain' in symptoms or 'abdominal pain' in symptoms:
        descriptions.append('Stomach pain can have many causes, including indigestion, infection, or stress. Severe or persistent pain should be evaluated by a doctor.')
        tips.extend([
            'Rest and avoid heavy meals.',
            'Apply a warm compress to the abdomen.',
            'Avoid foods that trigger pain.',
            'Seek medical help if pain is severe or accompanied by vomiting/fever.'
        ])
        medicines.extend([
            'Antacids (consult doctor)',
            'Paracetamol for pain (avoid NSAIDs unless prescribed)',
            'Probiotics (consult doctor)'
        ])
    if 'rash' in symptoms:
        descriptions.append('A rash is a noticeable change in the texture or color of your skin. It can be caused by allergies, infections, or other conditions.')
        tips.extend([
            'Keep the affected area clean and dry.',
            'Avoid scratching the rash.',
            'Use mild soap and avoid irritants.',
            'Consult a doctor if rash is widespread or persistent.'
        ])
        medicines.extend([
            'Antihistamine tablets or creams (consult doctor)',
            'Calamine lotion',
            'Hydrocortisone cream (consult doctor)'
        ])
    if 'shortness of breath' in symptoms:
        descriptions.append('Shortness of breath is difficulty breathing or feeling unable to get enough air. It can be a sign of a serious condition and may require urgent care.')
        tips.extend([
            'Sit upright and try to relax.',
            'Loosen tight clothing.',
            'Seek immediate medical attention if severe or accompanied by chest pain.'
        ])
        medicines.extend([
            'Inhalers (for asthma, consult doctor)',
            'Bronchodilators (consult doctor)'
        ])
    if not tips:
        descriptions.append('No specific symptoms detected. Maintain a healthy lifestyle and consult a doctor for any health concerns.')
        tips.append('Maintain a healthy diet and get enough sleep.')
    if not medicines:
        medicines.append('Consult a healthcare professional for proper medication.')
    return descriptions, tips, medicines

# --- File Description Route ---
@app.route('/describe/<filename>', methods=['GET', 'POST'])
def describe_file(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    text = extract_text_from_file(filepath)
    # Dynamically set max_length for summarizer based on input length
    if text:
        input_length = len(text.split())
        if input_length < 10:
            max_length = 4  # For very short inputs, keep summary very short
            min_length = 2
        else:
            max_length = min(120, max(30, int(input_length * 0.5)))
            min_length = 15
        description = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
    else:
        description = 'No readable text found.'
    symptom_descriptions = []
    health_tips = []
    medicines = []
    clinics = []
    user_location = ''
    if request.method == 'POST':
        symptoms = request.form.get('symptoms', '')
        user_location = request.form.get('location', '')
        symptom_descriptions, health_tips, medicines = check_symptoms(symptoms)
        clinics = find_nearby_clinics(user_location)
    # If GET and user_location is not set, try to get from query param or default
    elif request.method == 'GET':
        user_location = request.args.get('location', '')
        if user_location:
            clinics = find_nearby_clinics(user_location)
    return render_template('description.html', filename=filename, description=description, symptom_descriptions=symptom_descriptions, health_tips=health_tips, medicines=medicines, clinics=clinics, user_location=user_location, api_key=GOOGLE_MAPS_API_KEY)

# --- Google Maps/Places API for Clinics ---
GOOGLE_MAPS_API_KEY = 'AIzaSyAi2zJcZ_OEEoyIvIJqg8SdrzNWlaprMZY'  # Updated with your new API key

def find_nearby_clinics(location):
    # Use Google Places API to find clinics/hospitals near the location
    url = f'https://maps.googleapis.com/maps/api/place/textsearch/json?query=clinic+hospital+near+{location}+India&key={GOOGLE_MAPS_API_KEY}'
    try:
        response = requests.get(url)
        data = response.json()
        clinics = []
        for result in data.get('results', [])[:10]:
            clinics.append({
                'name': result['name'],
                'address': result['formatted_address'],
                'distance': 'N/A',
                'lat': result['geometry']['location']['lat'],
                'lng': result['geometry']['location']['lng']
            })
        return clinics
    except Exception as e:
        return []

@app.route('/get_location', methods=['POST'])
def get_location():
    data = request.get_json()
    location = data.get('location', '')
    clinics = find_nearby_clinics(location)
    return jsonify({'clinics': clinics})

# --- Profile, History, About ---
@app.route('/profile')
def profile():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('profile.html', username=session['username'])

@app.route('/history')
def history():
    # Placeholder: You can implement user upload history here
    return render_template('history.html', history=[])

@app.route('/about')
def about():
    return render_template('about.html')

# --- Medicine Summary Route ---
from transformers import pipeline as hf_pipeline

# Use a biomedical NER model for better medicine detection
ner = hf_pipeline('ner', model='d4data/biomedical-ner-all', aggregation_strategy='simple')


# --- Extract Medicines from Handwritten Prescription ---
def extract_medicines_from_text(text):
    """
    Use biomedical NER to extract medicine names from OCR text.
    Removes duplicates and near-duplicates (case-insensitive, whitespace-insensitive, fuzzy match).
    """
    entities = ner(text)
    medicines = []
    for ent in entities:
        if ent['entity_group'].lower() == 'chemical':
            med = ent['word'].strip().lower()
            med_norm = re.sub(r'\s+', '', med)
            # Fuzzy deduplication: skip if similar to any already in list
            is_duplicate = False
            for existing in medicines:
                existing_norm = re.sub(r'\s+', '', existing)
                ratio = SequenceMatcher(None, med_norm, existing_norm).ratio()
                if ratio > 0.8:  # 80% similarity threshold
                    is_duplicate = True
                    break
            if not is_duplicate:
                medicines.append(med)
    return medicines

# --- Ask AI about a medicine if not in base ---
# Local text2text-generation pipeline for medicine Q&A
# Use an even smaller, quantized model for maximum speed
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def get_fast_pipeline():
    model_name = "google/flan-t5-small"
    # Try to use a quantized version if available
    try:
        # Import inside try block to avoid ImportError if optimum is not installed
        from optimum.intel.openvino import OVModelForSeq2SeqLM
        model = OVModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    except Exception:
        # Fallback to normal pipeline
        return pipeline(
            "text2text-generation",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            model_kwargs={"low_cpu_mem_usage": True} if torch.cuda.is_available() else {},
        )

local_qa = get_fast_pipeline()

def ask_ai_about_medicine(medicine_name):
    prompt = (
        f"What is {medicine_name}? What is it used for? "
        "List at least 7 different uses in bullet points (each bullet should be a unique use, not a rewording), and then give a very detailed summary (at least 7 sentences, avoid repeating the bullet points) on separate lines."
    )
    # logging.info(f"[LOCAL HF] Prompt: {prompt}")
    try:
        result = local_qa(prompt, max_new_tokens=768)  # Further increased max_new_tokens for even longer answers
        answer = result[0]['generated_text']
        # logging.info(f"[LOCAL HF] Answer: {answer}")
        # Try to split into uses (bullets) and summary
        lines = [line.strip() for line in answer.split('\n') if line.strip()]
        uses = []
        summary_lines = []
        in_bullets = False
        for line in lines:
            if line.startswith('-') or line.startswith('•'):
                uses.append(line)
                in_bullets = True
            elif in_bullets and (line.startswith('-') or line.startswith('•')):
                uses.append(line)
            elif in_bullets and not (line.startswith('-') or line.startswith('•')):
                summary_lines.append(line)
            elif not in_bullets and (line.lower().startswith('use:') or line.lower().startswith('uses:')):
                in_bullets = True
            elif not in_bullets:
                summary_lines.append(line)
        # If no bullets detected, try to split by 'Use:' or 'Summary:'
        if not uses:
            if 'use:' in answer.lower() and 'summary:' in answer.lower():
                parts = re.split(r'(?i)summary:', answer, maxsplit=1)
                uses_part = parts[0].replace('Use:', '').replace('Uses:', '').strip()
                uses = [f"- {u.strip()}" for u in uses_part.split('.') if u.strip()]
                summary = parts[1].strip() if len(parts) > 1 else ''
            else:
                uses = [answer.strip()]
                summary = ''
        else:
            summary = '\n'.join(summary_lines).strip()
        return {'use': uses, 'summary': summary}
    except Exception as e:
        logging.error(f"[LOCAL HF] Error for {medicine_name}: {e}")
        return {'use': ['No info'], 'summary': 'No summary', 'error': str(e)}

@app.route('/medicine_summary/<filename>')
def medicine_summary(filename):
    # Extract text from the uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    text = extract_text_from_file(filepath)
    # Use NER to extract medicine names (fallback to filtered n-grams if none found)
    medicines = extract_medicines_from_text(text)
    if not medicines:
        import re
        # Replace newlines and commas with spaces, then remove other non-alphanumeric chars
        norm_text = re.sub(r'[\n,]', ' ', text.lower())
        norm_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', norm_text)
        words = norm_text.split()
        ngrams = set()
        for n in range(1, 4):
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i + n])
                # Only consider n-grams that look like medicine names (start with uppercase or are long enough)
                if (ngram and (ngram[0].isupper() or len(ngram) > 5)):
                    ngrams.add(ngram.strip())
        # Remove common stopwords, newlines, and commas
        stopwords = set([
            'and', 'or', 'the', 'for', 'with', 'to', 'of', 'in', 'on', 'by', 'is', 'a', 'an', 'as', 'at', 'from',
            'that', 'this', 'it', 'be', 'are', 'was', 'were', 'has', 'have', 'had', 'not', 'but', 'if', 'then', 'so',
            'do', 'does', 'did', 'can', 'will', 'would', 'should', 'may', 'might', 'must', 'could', '', '\n', ',',
        ])
        medicines = [ng for ng in ngrams if ng not in stopwords and len(ng) > 2]
    med_infos = []
    for med in medicines:
        try:
            ai_response = ask_ai_about_medicine(med)
        except Exception as e:
            logging.error(f"Error querying OpenAI for {med}: {e}")
            ai_response = {'error': str(e)}
        # Show the raw AI response for debugging
        content = str(ai_response)
        med_infos.append({'name': med.title(), 'ai_response': content})
    return render_template('medicine_summary.html', filename=filename, medicine_infos=med_infos)

# Set OpenAI API key for the project
import openai
openai.api_key = os.environ.get("OPENAI_API_KEY")

# --- Google Cloud Vision API Setup ---
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\cmsh8\medical-ai-assistant\gcloud-vision-key.json'

# --- Enhanced Text Extraction with Google Cloud Vision ---
def extract_text_with_vision(filepath):
    client = vision.ImageAnnotatorClient()
    with open(filepath, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    if not texts:
        return ''
    # Combine all parts of the detected text
    full_text = ' '.join([text.description for text in texts])
    return full_text

# --- Update extract_text_from_file to use Google Vision API if needed ---
def extract_text_from_file(filepath, use_google_vision=False):
    ext = filepath.rsplit('.', 1)[1].lower()
    text = ''
    if ext in ['png', 'jpg', 'jpeg']:
        client = vision.ImageAnnotatorClient()
        with io.open(filepath, 'rb') as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        response = client.text_detection(image=image)
        texts = response.text_annotations
        if texts:
            text = texts[0].description
        # If low confidence in OCR result, fallback to Google Vision API
        if use_google_vision and len(text) < 10:
            text = extract_text_with_vision(filepath)
    elif ext == 'pdf':
        doc = fitz.open(filepath)
        for page in doc:
            text += page.get_text()
    return text

@app.route('/prescription', methods=['POST'])
def handle_prescription():
    file = request.files.get('file')
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    # OCR: Extract text from image/pdf using Google Vision API
    text = extract_text_from_file(filepath, use_google_vision=True)
    # Extract medicine names
    medicines = extract_medicines_from_text(text)
    medicine_infos = []
    for med in medicines:
        # Always ask OpenAI, do not check the knowledge base
        ai_response = ask_ai_about_medicine(med)
        info = {
            'use': ai_response.get('use', 'No info'),
            'summary': ai_response.get('summary', 'No summary'),
            'ai_response': ai_response  # Include full AI response for display
        }
        medicine_infos.append({'name': med, **info})
    return jsonify({'medicines': medicine_infos})

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

if __name__ == '__main__':
    app.run(debug=True)
