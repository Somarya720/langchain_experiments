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
prompt_translate = RunnableLambda(lambda output: prompt_translate_template.format_prompt(text=output, **translation_data))

# create the chain
chain = prompt_fact_template | model | StrOutputParser() | prompt_translate | model | StrOutputParser()

# invoke the chain
response = chain.invoke(fact_data)

print(response)

# translation_data = {
#     'text': fact_response.content,
#     'language': 'Portugal'
# }

# fact_prompt_template = ChatPromptTemplate.from_messages(fact_template)
# fact_prompt = fact_prompt_template.invoke(fact_data)
# fact_response = model.invoke(fact_prompt)

# translate_prompt_template = ChatPromptTemplate.from_messages(translate_template)
# translate_prompt = translate_prompt_template.invoke(translation_data)
# translation_response = model.invoke(translate_prompt)

# print(fact_response.content, translation_response.content, sep='\n\n')
