from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

llm = ChatOpenAI(model='gpt-4.1-nano')
parser = StrOutputParser()

# invoke llm
ai_response = llm.invoke("Who won the world cup in 2014?")
# parse the response
response = parser.invoke(ai_response)

print(response)