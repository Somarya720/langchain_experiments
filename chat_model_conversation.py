from langchain_core.messages import SystemMessage,  HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

messages = [
    SystemMessage("You are an expert in social media content strategy"),
    HumanMessage("Give a short tip to create engaging postson instagram")
]

llm = ChatOpenAI(model='gpt-3.5-turbo')

result = llm.invoke(messages)

print(result.content)


