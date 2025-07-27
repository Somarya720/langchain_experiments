from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4o-mini')

review_template = [
    ("system", "You are a feedback analysing expert"),
    ("human", "Analyze the feedback {review} and classify it as positive, negative, neutral and escalate to human agent.")
]

positive_review = [
    ("system", "You are a feedback analysing expert"),
    ("human", "Send a thankful response based on the feedback: {review}")
]

negative_review = [
    ("system", "You are a feedback analysing expert"),
    ("human", "Send an apologizing response based on the feedback: {review}")
]

neutral_review = [
    ("system", "You are a feedback analysing expert"),
    ("human", "Send a request for more details on the feedback: {review}")
]

escalate_review = [
    ("system", "You are a feedback analysing expert"),
    ("human", "Send an apologizing response based on the feedback: {review} and also mention that you are connecting them to a human agent")
]

# generate prompt templates
prompt_review_template = ChatPromptTemplate.from_messages(review_template)
prompt_positive_template = ChatPromptTemplate.from_messages(positive_review)
prompt_negative_template = ChatPromptTemplate.from_messages(negative_review)
prompt_neutral_template = ChatPromptTemplate.from_messages(neutral_review)
prompt_escalate_template = ChatPromptTemplate.from_messages(escalate_review)

# create branches
branch_chain = RunnableBranch(
    (
        lambda x: 'positive' in x,
        RunnableLambda(lambda x: {'review': x}) | prompt_positive_template | model | StrOutputParser()
    ),
    (
        lambda x: 'escalate' in x,
        RunnableLambda(lambda x: {'review': x}) | prompt_escalate_template | model | StrOutputParser()
    ),
    (
        lambda x: 'negative' in x,
        RunnableLambda(lambda x: {'review': x}) | prompt_negative_template | model | StrOutputParser()
    ),
    RunnableLambda(lambda x: {'review': x}) | prompt_neutral_template | model | StrOutputParser()
    
)

# Create chain
conditional_chain = prompt_review_template | model | StrOutputParser()
chain = conditional_chain | branch_chain

review = "I'm not sure about the product yet. Can you tell me more about its features and benefits?"
# invoke llm
response = chain.invoke({
    'review': review
})

print(response)