import streamlit as st
from get_model import model

st.title("Response Generator")
st.write("Interact with our LLM: ")

prompt = st.text_input("Write your prompts here..")

if st.button("Send"):
    result = model.invoke(prompt)
    st.write(result.content)