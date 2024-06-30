import streamlit as st
import nltk
nltk.download('stopwords')

import time
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

import io


def extract_text_from_pdf(file):
    with io.BytesIO(file.read()) as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text



def preprocess_text(text):
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    filtered_tokens = [ps.stem(w) for w in tokens if not w in stop_words]
    preprocessed_text = ' '.join(filtered_tokens)
    return preprocessed_text

def main():
    st.title("Resume Scorecard")
    st.write("Enter the Key Job Specifications and upload your resume to see the matching score.")

    # Job requirements input
    job_requirements = st.text_area("Job Specifications")

    # Candidate's resume input
    resume_file = st.file_uploader("Upload your Resume", type=["pdf"])

    if st.button("Match") and resume_file is not None:
        # Preprocess the job requirements
        job_description = preprocess_text(job_requirements)

        # Extract text from the uploaded resume
        resume_text = extract_text_from_pdf(resume_file)
        # Preprocess the resume text
        resume = preprocess_text(resume_text)

        # Calculate TF-IDF
        corpus = [job_description, resume]
        tfidf = TfidfVectorizer().fit_transform(corpus)

        # Calculate cosine similarity
        cosine_similarities = cosine_similarity(tfidf[0:1], tfidf[1:2]).flatten()
        score = cosine_similarities[0]

       # Output result
        score_text = "The candidate's resume matches the job requirements with a score of {:.2%}.".format(score)
        st.write(score_text)

        # Delay for 2 seconds
        time.sleep(2)

 
        if score >= 0.10:
            st.success("Congratulations! Your resume matches the job requirements")
            time.sleep(1)
            st.balloons()
            time.sleep(1)
            st.write("Here are some job opportunities for you.")
          
        elif score < 0.10 and score >= 0.01:
            st.warning("Your resume partially matches the job requirements. Consider further improving your skills and experience.")
            st.write("Here are some courses for you:")
        else:
            st.error("Oops! Your resume does not match the job requirements. We recommend exploring relevant courses to enhance your qualifications.")
            st.write("Here are some courses for you:")

if __name__ == '__main__':
    main()