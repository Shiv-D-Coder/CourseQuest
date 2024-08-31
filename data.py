import requests
from bs4 import BeautifulSoup
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# URL to scrape
url = "https://brainlox.com/courses/category/technical"

# Fetch the data from the URL
response = requests.get(url)

# Check if the request was successful
if response.status_code != 200:
    raise Exception(f"Failed to retrieve data: {response.status_code}")

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.content, "html.parser")

# Find all course boxes on the page
course_boxes = soup.select('div.single-courses-box')

# Extract courses
courses = []
for box in course_boxes:
    title_tag = box.select_one('div.courses-content h3')
    description_tag = box.select_one('div.courses-content p')

    title = title_tag.text.strip() if title_tag else "No title found"
    description = description_tag.text.strip() if description_tag else "No description found"

    courses.append({
        "title": title,
        "description": description
    })

# Save extracted courses to a JSON file
with open("courses_data.json", "w", encoding="utf-8") as f:
    json.dump(courses, f, ensure_ascii=False, indent=4)

# Create embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
texts = [f"{course['title']} {course['description']}" for course in courses]
embeddings = model.encode(texts)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype(np.float32))

# Save FAISS index
faiss.write_index(index, "course_embeddings.index")

print("Data and embeddings have been saved.")
