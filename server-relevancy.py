from flask import Flask, request, jsonify
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_glove_model(glove_file):
    print("Loading GloVe Model...")
    glove_model = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]])
            glove_model[word] = embedding
    print("Done.", len(glove_model), "words loaded!")
    return glove_model

def vectorize_text(text, glove_model):
    words = text.split()
    vectors = [glove_model[word] for word in words if word in glove_model]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(50)  

def compute_similarity(vec1, vec2):
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]

glove_file = 'Models/glove.6B.300d.txt'
glove_model = load_glove_model(glove_file)

app = Flask(__name__)

@app.route('/calculate_score', methods=['POST'])
def calculate_score():
    try:

        data = request.json
        expert_relevancy_score = float(data['expert_relevancy_score'])
        candidate_relevancy_score = float(data['candidate_relevancy_score'])
        
        expert_background = data['expert_background']
        candidate_background = data['candidate_background']
    
        expert_background_vector = vectorize_text(expert_background, glove_model)
        candidate_background_vector = vectorize_text(candidate_background, glove_model)
     
        background_similarity_score = compute_similarity(expert_background_vector, candidate_background_vector)

        final_relevancy_score = (0.4 * expert_relevancy_score +
                                 0.4 * candidate_relevancy_score +
                                 0.2 * background_similarity_score)

        return jsonify({
            'final_relevancy_score': round(final_relevancy_score, 2),
            'background_similarity_score': round(background_similarity_score, 2)
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
