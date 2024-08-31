# Course Search Engine

A course search engine built with Flask, FAISS, and Sentence Transformers, enabling efficient retrieval of relevant courses based on user input.

## Tech Stack

- **Flask**: A lightweight web framework for building the API.
- **FAISS**: A library for efficient similarity search and clustering of dense vectors.
- **Sentence Transformers**: A model for encoding text into embeddings for semantic search.

## Features

- **Semantic Search**: Finds relevant courses based on user queries using vector embeddings.
- **Fast Retrieval**: Utilizes FAISS for quick and accurate search results.
- **User-Friendly**: Simple web interface for querying courses.

## Setup

1. **Clone the Repository**

    ```bash
    git clone https://github.com/yourusername/course-search-engine.git
    cd course-search-engine
    ```

2. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3. **Download Pre-trained Model and Index**

    - Ensure you have the `course_embeddings.index` file in the project directory.
    - Download the Sentence Transformers model used for encoding.

4. **Run the Application**

    ```bash
    python app.py
    ```

5. **Access the Application**

    Open your web browser and navigate to `http://127.0.0.1:5000` to use the search engine.

## Usage

- Enter your query in the search bar and click "Submit."
- View the list of relevant courses based on your input.

## Contributing

Feel free to open issues or submit pull requests to improve the project!
