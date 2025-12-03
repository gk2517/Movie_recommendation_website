<h1 align="center">Movie Recommender System 🎬</h1>

Unlock Your Next Favorite Film!
An NLP-powered recommendation engine that delivers personalized movie suggestions based on cast, genres, keywords, and production companies — all wrapped in a clean, Streamlit-based UI.

<h2>Project Overview</h2>

The Movie Recommender System leverages Python + Natural Language Processing (NLP) to help users discover movies tailored to their interests.
The preprocess.py file acts as the system’s data enrichment pipeline by pulling cast details, posters, descriptions, 
genres, and poster URLs directly from the TMDB API..

Using the Bag-of-Words model and cosine similarity, the system identifies how closely movies relate to each other based on:

Cast

Genres

Production companies

Tags and descriptions

You can also explore detailed metadata and access a full list of movies inside the app.

<h2>Sample Screenshots:</h2>
Recommendation Page

Showing personalized movie suggestions based on similarity.

<img width="1919" height="910" alt="Screenshot 2025-12-03 143626" src="https://github.com/user-attachments/assets/6cf7265f-a0d1-4dcd-a64f-bb959266954c" />
<img width="1919" height="910" alt="Screenshot 2025-12-03 143654" src="https://github.com/user-attachments/assets/8f45b1be-b891-4cdb-b52f-4ef98168e39a" />

Movie Description Page

Dive into cast details, genre, overview, and other key metadata.

<img width="1919" height="903" alt="Screenshot 2025-12-03 143721" src="https://github.com/user-attachments/assets/747dbf17-4cc5-49a6-aa76-43b8057e64c9" />

<img width="1919" height="773" alt="Screenshot 2025-12-03 143734" src="https://github.com/user-attachments/assets/54effa64-c0d5-49e1-a1d0-94e342f9a69f" />

All Movies Page

Navigate the entire movie list using buttons and sliders.

<img width="1919" height="895" alt="Screenshot 2025-12-03 143756" src="https://github.com/user-attachments/assets/38345e9b-f2de-4d00-96f5-c035e9fe7bbd" />

<h2>Installation Guide</h2> 

Follow these steps to set up and run the project locally:

1️. Clone the Repository

2️. Create a Virtual Environment

(Recommended for dependency isolation.)

3️. Install Dependencies: 
pip install -r requirements.txt

4️. Run the Application :
streamlit run main.py


⚠️ Note: On first run, the system may take a moment to generate required files and initialize the environment.

 <h2>Tech Stack</h2>

Python

Streamlit

NLP (Bag-of-Words & Cosine Similarity)

Pandas / NumPy

scikit-learn
