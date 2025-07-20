from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model='gpt-3.5-turbo')

messages_template_list = [
    ('system', 'You are a subject matter expert in {subject}'),
    ('human', 'How to create a {object} in {subject}')
]

# create template object from template_messages_list
template = ChatPromptTemplate.from_messages(messages_template_list)
prompt = template.invoke({
    'subject': 'c#',
    'object': 'function'
})

response = llm.invoke(prompt)
print(response.content)