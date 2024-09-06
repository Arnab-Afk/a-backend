from flask import Flask, request, jsonify
import PyPDF2
import re
import numpy as np
from numpy.linalg import norm
from gensim.models import Doc2Vec
import os

app = Flask(__name__)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub('[^a-z]', ' ', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = ' '.join(text.split())  # Remove extra spaces
    return text

# Endpoint to upload resume and process JD
@app.route('/match', methods=['POST'])
def match_resume_to_jd():
    # Check if the post request has the file part
    if 'resume' not in request.files:
        return jsonify({"error": "Resume file is required"}), 400
    
    file = request.files['resume']
    
    # Check if the file is a PDF
    if file.filename == '' or not file.filename.endswith('.pdf'):
        return jsonify({"error": "A valid PDF file is required"}), 400
    
    # Load the resume PDF
    try:
        pdf = PyPDF2.PdfReader(file)
        resume = ""
        for i in range(len(pdf.pages)):
            page_obj = pdf.pages[i]
            resume += page_obj.extract_text()
    except Exception as e:
        return jsonify({"error": f"Error processing PDF file: {str(e)}"}), 500
    
    # Preprocess resume text
    input_cv = preprocess_text(resume)
    
    # Get JD from the request body
    jd = request.form.get('jd')
    if not jd:
        return jsonify({"error": "Job description (JD) is required"}), 400
    
    # Preprocess JD
    input_jd = preprocess_text(jd)
    
    # Load the Doc2Vec model
    try:
        model = Doc2Vec.load('Models/cv_job_maching.model')
    except Exception as e:
        return jsonify({"error": f"Error loading model: {str(e)}"}), 500
    
    # Infer vectors
    v1 = model.infer_vector(input_cv.split())
    v2 = model.infer_vector(input_jd.split())
    
    # Calculate similarity
    similarity = 100 * (np.dot(np.array(v1), np.array(v2))) / (norm(np.array(v1)) * norm(np.array(v2)))
    
    # Return the similarity score as JSON
    return jsonify({"similarity": round(similarity, 2),
                    "resume_str":resume})

if __name__ == '__main__':
    app.run(debug=True)
