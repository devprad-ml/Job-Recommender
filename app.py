import streamlit as st
import pandas as pd
from recommender import build_recommender, recommend_jobs

@st.cache_data
def load_recommender(csv_path):
    df = pd.read_csv(csv_path)
    return build_recommender(df)

def main():
    st.title("Job Recommender System")

    uploaded_file = st.file_uploader("Upload cleaned jobs CSV", type=["csv"])
    if uploaded_file is not None:
        df, vectorizer, tfidf_matrix = load_recommender(uploaded_file)
        st.success("Recommender is ready!")

        st.subheader("Get Job Recommendations")
        job_title = st.text_input("Job Title")
        job_desc = st.text_area("Job Description")

        if st.button("Recommend Jobs"):
            if job_title and job_desc:
                results = recommend_jobs(df, vectorizer, tfidf_matrix, job_title, job_desc)
                st.write("### Top Job Recommendations:")
                for idx, row in results.iterrows():
                    st.markdown(f"**{row['Title']}**")
                    st.markdown(f"{row['JobDescription'][:300]}...")
                    st.markdown("---")
            else:
                st.warning("Please provide both title and description.")

if __name__ == "__main__":
    main()
