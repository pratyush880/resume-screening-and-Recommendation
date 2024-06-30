import streamlit as st
import nltk
from PyPDF2 import PdfReader
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import zipfile
from io import BytesIO

def main():
    st.title("Resume Ranker")
    st.write("Enter the job description and upload a resumes folder (in ZIP format) containing up to 50 resumes (in PDF format) to see the ranked resumes.")

    # Job requirements input
    job_description = st.text_area("Job Description")

    # Resumes folder input
    resumes_folder = st.file_uploader("Upload Resumes Folder", type=["zip"], accept_multiple_files=False)

    # Check if a resumes folder is uploaded
    if not resumes_folder:
        st.write("Please upload a resumes folder.")
        return

    if st.button("Rank Resumes"):
        # Extract the resumes from the uploaded folder
        zip_file = zipfile.ZipFile(resumes_folder)
        resume_data = []
        for rank, file_name in enumerate(zip_file.namelist(), start=1):
            if not file_name.lower().endswith(".pdf"):
                continue
            resume = zip_file.read(file_name)
            pdf_reader = PdfReader(BytesIO(resume))
            resume_text = ""
            for page in pdf_reader.pages:
                resume_text += page.extract_text()

            resume_name = file_name
            resume_data.append((rank, resume_text, resume_name))

        # Limit to the first 50 resumes if more than 50 are uploaded
        resumes = resume_data[:50]

        # Preprocess the job description
        ps = PorterStemmer()
        stop_words = set(stopwords.words('english'))
        job_description_tokens = word_tokenize(job_description.lower())
        job_description_filtered_tokens = [ps.stem(w) for w in job_description_tokens if not w in stop_words]

        # Rank resumes based on the matching score with job description
        ranked_resumes = rank_resumes(job_description_filtered_tokens, resumes)

        # Display ranked resumes in a table
        ranked_resumes = [(rank, resume_name, score) for rank, score, resume_name in ranked_resumes]
        df = pd.DataFrame(ranked_resumes, columns=["Rank", "Resume Name", "Matching Score"])
        df.set_index("Rank", inplace=True)
        st.write("Ranked Resumes:")
        st.table(df)

def rank_resumes(job_description_tokens, resume_data):
    scores = []
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    for rank, resume_text, resume_name in resume_data:
        resume_tokens = word_tokenize(resume_text.lower())
        resume_filtered_tokens = [ps.stem(w) for w in resume_tokens if not w in stop_words]
        score = sum(1 for token in resume_filtered_tokens if token in job_description_tokens)
        scores.append((rank, score, resume_name))

    ranked_resumes = sorted(scores, key=lambda x: x[1], reverse=True)
    return ranked_resumes

if __name__ == '__main__':
    main()
