import streamlit as st
from get_model import model
from langchain_core.prompts.loading import load_prompt
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
template_path = os.path.join(current_dir, 'prompt.json')

template = load_prompt(template_path)

st.title("Research analysis")
st.write("Get Summary of a research paper from our LLM")

paper_name = st.selectbox(
    "Enter the research paper",
    ("Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"),
    index=None,
    placeholder='select paper'
)

summary_type = st.selectbox(
    "Enter the type of summary",
    ("Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical")
)

length = st.selectbox(
    "Enter the length of the summary",
    ("short (1-2 paragraph)", "medium(3-4 paragraph)", "long(5-6 paragraph)")
)

if st.button("Summarize"):
    prompt = template.invoke(
        {
            'paper_input':paper_name,
            'length_input':length,
            'style_input':summary_type
        }
    )
    result = model.invoke(prompt)
    st.write(result.content)