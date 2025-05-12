import streamlit as st
from streamlit_chat import ComplaintMatcher
import pandas as pd

@st.cache_resource
def load_matcher():
    return ComplaintMatcher("Final_student_sentiment_cleaned.csv")  

matcher = load_matcher()

st.title("🎓 Student Complaints Assistant")
st.markdown("Enter your complaint below. The system will analyze and suggest a resolution.")

# Initialize session state
if "complaint" not in st.session_state:
    st.session_state.complaint = ""
if "response" not in st.session_state:
    st.session_state.response = []
if "show_feedback" not in st.session_state:
    st.session_state.show_feedback = False
if "genre" not in st.session_state:
    st.session_state.genre = ""

# Text area for user input
user_input = st.text_area("✍️ Your Complaint:", height=150, value=st.session_state.complaint)

# Button to process complaint
if st.button("🔍 Get Resolution"):
    if user_input.strip() == "":
        st.warning("Please enter a complaint before submitting.")
    else:
        with st.spinner("Processing your complaint..."):
            st.session_state.complaint = user_input
            response = matcher.get_response(user_input)
            st.session_state.response = response
            st.session_state.genre = response[0]['genre'] if response else ""
            st.session_state.show_feedback = True

# Show result if previously computed
if st.session_state.show_feedback and st.session_state.response:
    st.subheader("📋 Response:")
    for res in st.session_state.response:
        st.markdown(f"**Matched Complaint**: {res['matched_complaint']}")
        st.markdown(f"**Suggested Resolution**: {res['resolution']}")
        st.markdown(f"**Similarity**: {res['similarity']}")
        st.markdown("---")

    st.subheader("🤔 Are you satisfied with the suggested solution?")
    feedback = st.radio("Your feedback:", ["Satisfied", "Neutral", "Dissatisfied"], key="feedback_radio")

    if feedback in ["Satisfied", "Neutral"]:
        st.success("Thank you for your feedback! 😊")
    else:
        st.warning("Showing better alternative solutions...")
        alt_df = matcher.get_alternative_solutions(
            corrected_input=st.session_state.response[0]["corrected_input"],
            genre=st.session_state.genre,
            top_n=2
        )
        if not alt_df:
            st.error("Sorry, no alternative solutions found.")
        else:
            for alt in alt_df:
                st.markdown(f"**Alternative Report** (Similarity: {alt['cross_similarity']:.2f}, Satisfaction: {alt['satisfaction_score']:.2f}):")
                st.markdown(f"- **Original Report:** {alt['Reports_clean']}")
                st.markdown(f"- **Suggested Resolution:** {alt['Resolution_clean']}")
