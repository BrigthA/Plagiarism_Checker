import streamlit as st
from sentence_transformers import SentenceTransformer, util

# Page configuration
st.set_page_config(
    page_title="Plagiarism Checker",
    page_icon="🔍",
    layout="centered"
)

# Load model with st.cache_resource so it only downloads/loads once
@st.cache_resource
def load_similarity_model():
    # Lightweight, high-performance model (~80MB)
    return SentenceTransformer("all-MiniLM-L6-v2")

with st.spinner("Loading lightweight semantic similarity model..."):
    model = load_similarity_model()

# App Header
st.title("🔍 Plagiarism Checker")
st.caption("Powered by Streamlit and Sentence-Transformers running locally.")

# Input areas
st.markdown("### Paste Texts to Compare")
source_text = st.text_area(
    "Original Source Text", 
    placeholder="Paste the original reference text here...",
    height=150
)

suspect_text = st.text_area(
    "Suspect / Submission Text", 
    placeholder="Paste the text you want to check for plagiarism here...",
    height=150
)

# Check button
if st.button("Check Similarity", type="primary"):
    if not source_text.strip() or not suspect_text.strip():
        st.warning("Please provide both source and suspect texts.")
    else:
        with st.spinner("Analyzing semantic similarity..."):
            # Encode texts to get vector embeddings
            embedding1 = model.encode(source_text, convert_to_tensor=True)
            embedding2 = model.encode(suspect_text, convert_to_tensor=True)

            # Compute cosine similarity score (0.0 to 1.0)
            score = util.cos_sim(embedding1, embedding2).item()
            percentage = score * 100

        # Display Results
        st.divider()
        st.subheader("Analysis Results")
        
        # Metric display
        st.metric(label="Semantic Similarity Score", value=f"{percentage:.2f}%")
        
        # Visual progress bar
        st.progress(score)

        # Verdict logic
        if score > 0.8:
            st.error("🚨 **High Similarity Detected:** Likely plagiarized or heavily paraphrased.")
        elif score > 0.5:
            st.warning("⚠️ **Moderate Similarity:** Some overlapping phrasing or concepts found.")
        else:
            st.success("✅ **Low Similarity:** Texts appear to be largely original or unrelated.")