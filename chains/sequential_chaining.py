from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Tasks: format the template, invoke the llm, extract the output, use it in translation template, invoke llm, extract the output

model = ChatOpenAI(model='gpt-4.1-nano')

# Prepare the templates
fact_template = [
    ("system", "You are a sports expert in the field of {game}"),
    ("human", "Who won the {tournament} in {year}?")
]
translate_template = [
    ("system", "You are a translation expert"),
    ("human", "Translate the text {text} to {language}")
]

prompt_fact_template = ChatPromptTemplate.from_messages(fact_template)
prompt_translate_template = ChatPromptTemplate.from_messages(translate_template)

# Prepare the prompt parameters
fact_data = {
    'game': 'football',
    'tournament': 'World cup',
    'year': 2022
}
translation_data = {
    'language': 'French'
}

# create runnable lambda
prompt_translate_params = RunnableLambda(lambda output: {**translation_data, 'text': output})

# create the chain
chain = prompt_fact_template | model | StrOutputParser() | prompt_translate_params | prompt_translate_template | model | StrOutputParser()

# invoke the chain
response = chain.invoke(fact_data)

print(response)