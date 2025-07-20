from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Loads the secret key to the environment variable from .env
load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo")

result = llm.invoke("What is the square root of 49")

print(result, end="\n\n")

# Response message
print(f"Response message: {result.content}")