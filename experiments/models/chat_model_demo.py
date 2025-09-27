from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4.1-nano', temperature=0.2, max_completion_tokens=10)

result = model.invoke("Write a 5 line poem on Cricket.")

print(result.content)