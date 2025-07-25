from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# create model
llm = ChatOpenAI(model='gpt-4.1-nano')

template = [
    ("system", "You are an expert about {animal}"),
    ("human", "tell me {fact_count} facts")
]

# generating prompt from template 
prompt_template = ChatPromptTemplate.from_messages(template)
prompt = prompt_template.invoke({
    'animal': 'Elephant',
    'fact_count': 2
})

# generate ai response
llm_response = llm.invoke(prompt)

# get required output
response = llm_response.content

print(response)