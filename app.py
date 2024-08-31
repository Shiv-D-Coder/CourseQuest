from flask import Flask, request, jsonify
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json

app = Flask(__name__)

# Initialize the model and index
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("course_embeddings.index")  # Replace with your index file path

# Load course data from JSON file
def load_courses():
    try:
        with open('courses_data.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        return []

courses = load_courses()

@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/query', methods=['POST'])
def query():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    user_input = request.json.get('query', '')
    if not user_input:
        return jsonify({'error': 'Query is required'}), 400

    if index is None:
        return jsonify({'error': 'FAISS index is not loaded'}), 500

    # Encode the user input
    try:
        query_embedding = model.encode([user_input]).astype(np.float32)
    except Exception as e:
        print(f"Error encoding user input: {e}")
        return jsonify({'error': 'Error encoding query'}), 500

    # Perform the search
    try:
        _, indices = index.search(query_embedding, k=5)  # Adjust 'k' for the number of results
        indices = indices[0]  # Get the first (and only) array from the result
    except Exception as e:
        print(f"Error performing search: {e}")
        traceback.print_exc()  # Print the stack trace for debugging
        return jsonify({'error': 'Error during search'}), 500

    # Ensure indices are within bounds
    if len(courses) < len(indices):
        return jsonify({'error': 'Index out of range in course data'}), 500

    # Collect results
    results = []
    for i in indices:
        if i < len(courses):
            results.append(courses[i])
        else:
            print(f"Index {i} out of range in course data.")
            return jsonify({'error': 'Index out of range in course data'}), 500

    if not results:
        return jsonify({'message': 'No results found.'}), 200

    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)
