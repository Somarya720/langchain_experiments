from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

model = OpenAI(model='gpt-3.5-turbo-instruct') # gpt-3.5-turbo-instruct not a chatmodel, it is a llm

result = model.invoke("What is the capital of India?")

print(result)