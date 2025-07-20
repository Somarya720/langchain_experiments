from langchain_core.messages import SystemMessage,  HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# messages = [
#     SystemMessage("You are an expert in social media content strategy"),
#     HumanMessage("Give a short tip to create engaging postson instagram")
# ]

llm = ChatOpenAI(model='gpt-3.5-turbo')

messages = [
    SystemMessage("You are a bollywood movie suggestion expert"),
    HumanMessage("Suggest one good hindi movie"),
]

suggestion_response = llm.invoke(messages)
suggestion = suggestion_response.content

print(f'AI Suggestion: {suggestion}')

# Adding AI response
messages.extend([
    AIMessage(suggestion),
    HumanMessage("Name the actors of the movie")
])

actor_response = llm.invoke(messages)
print(f"Actors: {actor_response.content}")

# Get token count
print(suggestion_response.response_metadata['token_usage']['total_tokens'], actor_response.response_metadata['token_usage']['total_tokens'])