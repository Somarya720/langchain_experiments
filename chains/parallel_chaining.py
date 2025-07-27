from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableSequence
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4.1-nano')

summary_template = [
    ("system", "You are a movie critic"),
    ("human", "Give a summary on the movie {movie}")
]

plot_template = [
    ("system", "You are a movie critic"),
    ("human", "Tell 2 sentences about the plot of {summary}")
]

character_template = [
    ("system", "You are a movie critic"),
    ("human", "Describe 3 characters of {summary}")
]
summary_parameters = {'movie': 'Swades'}

prompt_summary_template = ChatPromptTemplate.from_messages(summary_template)
prompt_plot_template = ChatPromptTemplate.from_messages(plot_template)
prompt_character_template = ChatPromptTemplate.from_messages(character_template)

# create chains to be processed parallely
def chain_plot(template: ChatPromptTemplate) -> RunnableSequence:
    return RunnableLambda(lambda summary: {'summary': summary}) | template | model | StrOutputParser()

# create chain to handle parralel chains
chain = (
    prompt_summary_template 
    | model 
    | StrOutputParser() 
    | RunnableParallel(
        branches={
            'plot': chain_plot(prompt_plot_template),
            'characters': chain_plot(prompt_character_template)
        }
    ) 
    | RunnableLambda(
        lambda result: f"Plot: {result['branches']['plot']} \n\nCharacters: {result['branches']['characters']}"
    )
)

resposne = chain.invoke(summary_parameters)
print(resposne)