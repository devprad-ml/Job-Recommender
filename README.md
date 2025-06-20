# Job-Recommender

## Job Recommender System

A simple, content-based job recommendation system that suggests relevant job postings based on user-provided job title and description. Built using Python, Streamlit, and scikit-learn.

---

## Features

- Upload your own cleaned job postings CSV.
- Enter a job title and job description to receive tailored job recommendations.
- Uses TF-IDF vectorization and cosine similarity for content-based matching.
- Simple Streamlit web app interface.

---

## How It Works

### Data Preparation
- Use the included `data_setup.ipynb` notebook to clean and preprocess raw job posting data.
- This involves removing duplicates, cleaning text fields, and saving the processed data as `jobs.csv`.

### Running the App
- The main application (`app.py`) loads the cleaned job postings and initializes the recommender.
- Users upload the CSV file, input a job title and description, and receive the top matching job postings.

---

## File Overview

- `app.py`: Streamlit app for job recommendation.
- `recommender.py`: Core logic for building the job recommender and generating recommendations.
- `data_setup.ipynb`: Notebook for cleaning and preparing the job posting dataset.
- `jobs.csv`: Example of a cleaned dataset for use in the app.
- `data job posts.csv`: Raw dataset (very large file).

---


