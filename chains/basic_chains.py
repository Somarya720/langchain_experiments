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

# create prompt template
prompt_template = ChatPromptTemplate.from_messages(template)

# build chain
chain = prompt_template | llm 

# invoke the chain
response = chain.invoke(
    {
        'animal': 'elephant',
        'fact_count': 2
    }
)

print(response)