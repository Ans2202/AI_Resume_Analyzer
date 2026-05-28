import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# -------------------- NLTK Downloads --------------------

nltk.download("punkt")
nltk.download('punkt_tab')
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")  # Fix for NLTK 3.9+ on Streamlit Cloud

# -------------------- PAGE CONFIG --------------------

st.set_page_config(
    page_title="AI Resume Matcher",
    page_icon="📄",
    layout="wide"
)

# -------------------- CUSTOM CSS --------------------

st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

h1, h2, h3 {
    color: white;
}

.stButton>button {
    background: linear-gradient(90deg,#ff4b4b,#ff914d);
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

.stTextArea textarea {
    border-radius: 10px;
}

.stFileUploader {
    border-radius: 10px;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)

# -------------------- TITLE --------------------

st.title("AI Resume Job Match Analyzer")

st.markdown("""
<div style='font-size:18px; color:gray;'>

Upload your resume and compare it with a job description using NLP-based analysis.

• ATS Match Score  
• Missing Skills Detection  
• Keyword Analysis  
• Resume Suggestions  
• Technical Skills Detection  

</div>
""", unsafe_allow_html=True)

st.markdown("---")

# -------------------- SIDEBAR --------------------

with st.sidebar:

    st.header("About This Tool")

    st.info("""
This project helps you:

• Measure ATS Resume Match Score  
• Detect missing keywords  
• Analyze resume quality  
• Improve hiring chances  
• Compare resume with job role
""")

    st.header("How It Works")

    st.markdown("""
1. Upload Resume PDF  
2. Paste Job Description  
3. Click Analyze Match  
4. View Results
""")

# -------------------- FUNCTIONS --------------------

# PDF Reader
def extract_text_from_pdf(uploaded_file):

    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)

        text = ""

        for page in pdf_reader.pages:
            extracted = page.extract_text()

            if extracted:
                text += extracted

        return text

    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""


# Clean text
def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text


# Remove stopwords
def remove_stopwords(text):

    stop_words = set(stopwords.words('english'))

    words = word_tokenize(text)

    filtered_words = [word for word in words if word not in stop_words]

    return " ".join(filtered_words)


# Similarity calculation
def calculate_similarity(resume_text, job_description):

    resume_processed = remove_stopwords(clean_text(resume_text))

    job_processed = remove_stopwords(clean_text(job_description))

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(
        [resume_processed, job_processed]
    )

    score = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )[0][0] * 100

    return round(score, 2), resume_processed, job_processed


# Extract keywords
def extract_keywords(text, num_keywords=10):

    words = word_tokenize(text)

    words = [w for w in words if len(w) > 2]

    tagged_words = pos_tag(words)

    nouns = [
        w for w, pos in tagged_words
        if pos.startswith('NN') or pos.startswith('JJ')
    ]

    word_freq = Counter(nouns)

    return word_freq.most_common(num_keywords)


# Missing keywords
def get_missing_keywords(resume_text, job_text):

    resume_words = set(word_tokenize(clean_text(resume_text)))

    job_words = set(word_tokenize(clean_text(job_text)))

    missing = job_words - resume_words

    important_missing = [
        word for word in missing
        if len(word) > 3
    ]

    return list(important_missing)[:15]


# Technical skill detector
def detect_skills(text):

    tech_skills = [
        "python", "sql", "machine", "learning",
        "tensorflow", "excel", "powerbi",
        "tableau", "aws", "java", "react",
        "nodejs", "mongodb", "nlp",
        "streamlit", "pandas", "numpy",
        "scikit", "matplotlib"
    ]

    found = []

    for skill in tech_skills:

        if skill in text:
            found.append(skill)

    return found


# -------------------- MAIN APP --------------------

def main():

    uploaded_file = st.file_uploader(
        "Upload Your Resume (PDF)",
        type=['pdf']
    )

    job_description = st.text_area(
        "Paste Job Description",
        height=250
    )

    if st.button("Analyze Match"):

        if not uploaded_file or not job_description:

            st.warning("Please upload resume and paste job description.")

            return

        with st.spinner("Analyzing Resume..."):

            resume_text = extract_text_from_pdf(uploaded_file)

            if not resume_text:

                st.error("Could not extract text from PDF.")

                return

            # Similarity
            similarity_score, resume_processed, job_processed = calculate_similarity(
                resume_text,
                job_description
            )

            # Keywords
            resume_keywords = extract_keywords(resume_processed)

            job_keywords = extract_keywords(job_processed)

            # Missing Skills
            missing_keywords = get_missing_keywords(
                resume_text,
                job_description
            )

            # Detected Skills
            found_skills = detect_skills(resume_processed)

        # ---------------- RESULTS ----------------

        st.markdown("---")

        st.header("Analysis Results")

        # Score
        st.metric(
            label="ATS Match Score",
            value=f"{similarity_score}%"
        )

        # Progress bar
        st.progress(int(similarity_score))

        # Resume Strength
        if similarity_score >= 80:

            st.success("Strong Resume Match")

        elif similarity_score >= 60:

            st.info("Good Resume Match")

        else:

            st.warning("Resume Needs Improvement")

        # ---------------- PIE CHART ----------------

        st.subheader("Resume Match Visualization")

        fig, ax = plt.subplots()

        sizes = [similarity_score, 100 - similarity_score]

        labels = ['Matched', 'Missing']

        ax.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%'
        )

        ax.set_title("Resume vs Job Match")

        st.pyplot(fig)

        # ---------------- KEYWORDS ----------------

        st.subheader("Keyword Analysis")

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Resume Keywords")

            for word, count in resume_keywords:

                st.write(f"{word} ({count})")

        with col2:

            st.subheader("Job Keywords")

            for word, count in job_keywords:

                st.write(f"{word} ({count})")

        # ---------------- MISSING SKILLS ----------------

        st.subheader("Missing Keywords")

        if missing_keywords:

            st.write(", ".join(missing_keywords))

        else:

            st.success("Your resume covers most important keywords.")

        # ---------------- DETECTED SKILLS ----------------

        st.subheader("Detected Technical Skills")

        if found_skills:

            st.write(" | ".join(found_skills))

        else:

            st.warning("No major technical skills detected.")

        # ---------------- AI SUGGESTIONS ----------------

        st.subheader("Resume Suggestions")

        if similarity_score < 40:

            st.error("""
• Add more technical skills from the job description  
• Include more relevant projects  
• Improve ATS keyword optimization  
• Add measurable achievements  
""")

        elif similarity_score < 70:

            st.warning("""
• Improve ATS optimization  
• Add more frameworks/tools related to the role  
• Use stronger action verbs and achievements  
""")

        else:

            st.success("""
• Strong ATS optimization  
• Good alignment with job requirements  
• Resume is highly relevant for this role  
""")

# -------------------- RUN APP --------------------

main()
