# Simulate Langchain Expression Language for chains with functions

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# create the template
template = [
    ("system", "You are an expert in {sport}"),
    ("human", "Who won the {tournament} in {year}")
]

# Create the tasks in chain
model = ChatOpenAI(model='gpt-4.1-nano')
prompt_template = ChatPromptTemplate.from_messages(template)
parser = StrOutputParser()

# create the chain
chain = prompt_template | model | parser

response = chain.invoke({
    'sport': 'football',
    'tournament': 'World cup',
    'year': 2014
})

print(response)