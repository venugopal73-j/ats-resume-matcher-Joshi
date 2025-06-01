import streamlit as st
from resume_processor import extract_text
from matcher import get_match_score, analyze_keywords
import os

st.set_page_config(page_title="ATS Score Matcher", layout="centered")
st.title("📄 ATS Match Score Calculator")
st.write("Upload your resume and job description to get a match score out of 10.")

resume_file = st.file_uploader("📤 Upload Resume (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"])
jd_file = st.file_uploader("💼 Upload Job Description (PDF/TXT)", type=["pdf", "txt"])

if resume_file and jd_file:
    _, resume_ext = os.path.splitext(resume_file.name)
    _, jd_ext = os.path.splitext(jd_file.name)
    resume_ext = resume_ext[1:].lower()
    jd_ext = jd_ext[1:].lower()

    with st.spinner("Processing..."):
        try:
            resume_text = extract_text(resume_file, resume_ext)
            jd_text = extract_text(jd_file, jd_ext)
            score = get_match_score(resume_text, jd_text)
            st.success(f"✅ Match Score: **{score}/10**")

            keyword_analysis = analyze_keywords(resume_text, jd_text)
            missing_keywords = keyword_analysis["missing_keywords"]
            matched_keywords = keyword_analysis["matched_keywords"]

            if missing_keywords:
                st.warning(f"💡 Missing Keywords: {', '.join(missing_keywords)}")
                st.write("Add these keywords to your resume to improve the match score.")
            else:
                st.success("🎉 All JD keywords are present in your resume!")

            if matched_keywords:
                st.info(f"✅ Matched Keywords: {', '.join(matched_keywords)}")

            st.subheader("💡 Suggestions to Improve Your Resume:")
            st.markdown(
                """
                - **Tailor Your Resume:** Include missing keywords in relevant sections.
                - **Reorder Sections:** Prioritize sections that align with the JD.
                - **Quantify Achievements:** Use numbers and metrics to showcase impact.
                - **Use Action Verbs:** Start bullet points with strong action verbs.
                """
            )

        except Exception as e:
            st.error("An error occurred while processing the files.")
            st.exception(e)