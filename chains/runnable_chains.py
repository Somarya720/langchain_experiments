from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI(model='gpt-4.1-nano')

# create the template
template = [
    ("system", "You are an expert in {sport}"),
    ("human", "Who won the {tournament} in {year}")
]

prompt_template = ChatPromptTemplate.from_messages(template)

# runnable lambda to format the template
format_template = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
# runnable lambda to invoke the llm
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
# runnable lambda to extract only the content from ai response
extract_content = RunnableLambda(lambda x: x.content)

# create the chain
chain = RunnableSequence(first=format_template,middle=[invoke_model],last=extract_content)
# invoke the chain
response = chain.invoke({
    'sport': 'football',
    'tournament': 'World cup',
    'year': 2014
})

print(response)