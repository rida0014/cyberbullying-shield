import streamlit as st
from transformers import pipeline

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Cyberbullying Shield",
    layout="centered"
)

# ---------------------------------------------------
# Application Header
# ---------------------------------------------------
st.title("Real-Time AI Cyberbullying Detection Shield")

st.markdown(
    "### VTU Interdisciplinary Project Work | Course Code: 1BPRJ258"
)

st.write(
    "Type any comment or message below to test the platform's AI moderation system."
)

# ---------------------------------------------------
# Load AI Toxicity Detection Model
# ---------------------------------------------------
@st.cache_resource
def load_model():
    # Automatically downloads the highly targeted Toxic-BERT model 
    # directly into the web application memory.
    classifier = pipeline(
        "text-classification",
        model="unitary/toxic-bert"
    )
    return classifier

# Initialize AI Model
ai_shield = load_model()

# ---------------------------------------------------
# User Input Section
# ---------------------------------------------------
user_message = st.text_area(
    "User Comment Box:",
    placeholder="Write your text here...",
    height=120
)

# ---------------------------------------------------
# AI Prediction Logic
# ---------------------------------------------------
if st.button("Post Message to Timeline"):

    # Empty Input Validation
    if not user_message.strip():
        st.warning("Warning: Please enter a message to evaluate.")

    else:
        with st.spinner("Analyzing message intent characteristics..."):
            try:
                # AI Prediction Execution
                prediction = ai_shield(user_message)[0]

                label = prediction["label"]
                score = prediction["score"]

                # ---------------------------------------------------
                # Toxic Message Interception (Score > 0.70)
                # ---------------------------------------------------
                if score > 0.40:
                    st.error("CRITICAL INTERCEPTION ALERT: POST BLOCKED")
                    st.markdown(
                        f"""
                        **System Action:** Harmful content blocked automatically.

                        **Reason:** Toxic or cyberbullying language detected.

                        **Toxicity Probability:** {score * 100:.2f}%
                        """
                    )

                # ---------------------------------------------------
                # Safe Message Transmission Approval
                # ---------------------------------------------------
                else:
                    st.success("TRANSMISSION PERMITTED")
                    st.markdown(
                        f"""
                        **System Action:** Message approved successfully.

                        **Text Status:** Safe / Non-toxic

                        **Safety Probability:** {(1 - score) * 100:.2f}%
                        """
                    )

            except Exception as e:
                st.error(f"Prediction analysis engine failed: {e}")