import streamlit as st
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4.1-nano')

st.title("Response Generator")
st.write("Interact with our LLM: ")

prompt = st.text_input("Write your prompts here..")

if st.button("Send"):
    result = model.invoke(prompt)
    st.write(result.content)