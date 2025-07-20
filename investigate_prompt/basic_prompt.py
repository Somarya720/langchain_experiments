from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def invoke_llm_without_prompt(template: str, data: dict, model: ChatOpenAI) -> str:
    '''
    generates ai response from llm without using langchain prompt templates
    '''
    prompt = template.format(**data) # using inbuilt string format to add values to the template
    response = model.invoke(prompt)
    return response.content


llm = ChatOpenAI(model='gpt-3.5-turbo')

template = "Suggest {number} movies in {language} with genre {genre} released in {year} that would be fun to watch"
data_set = [
    {
        'number': 4,
        'language': 'hindi',
        'genre': 'comedy',
        'year': 2007
    }, 
    {
        'number': 3,
        'language': 'hindi',
        'genre': 'thriller',
        'year': 2008
    },
]

for data in data_set:
    response = invoke_llm_without_prompt(template, data, llm)
    print(response, end="\n-----------------------------\n")
