from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

prompt_template = PromptTemplate.from_template("Tell me a joke on {topic}", validate_template=False)
print(prompt_template.input_variables)

prompt = prompt_template.invoke({
    'topic': 'cricket'
})

print(prompt)

model = ChatOpenAI(model='gpt-4.1-nano')

result = model.invoke(prompt)
print(result.content)