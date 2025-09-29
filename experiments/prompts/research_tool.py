import streamlit as st

st.title("Research analysis")
st.write("Get Summary of a research paper from our LLM")

paper_name = st.selectbox(
    "Enter the research paper",
    ("a","b","c"),
    index=False,
    placeholder='select paper'
)

summary_type = st.selectbox(
    "Enter the type of summary",
    ("basic","mathematics-heavy", "analogy heavy")
)

length = st.selectbox(
    "Enter the length of the summary",
    ("short", "medium", "long")
)

if st.button("Summarize"):
    st.write("Summary")