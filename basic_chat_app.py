from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

# Loads the .env into environment variable
load_dotenv()

llm = ChatOpenAI(model='gpt-3.5-turbo')

chat_history = [
    SystemMessage(content="You are a helpful AI assistant")
]

while(True):
    user_msg = input("User: ")
    
    # quit the chat
    if user_msg.lower() == 'quit':
        print('You have ended the chat')
        break

    # Add user message
    chat_history.append(HumanMessage(content=user_msg))

    # Fetch ai response based on history of conversation
    ai_response = llm.invoke(chat_history)
    ai_msg = ai_response.content

    # Add ai message to chat history
    chat_history.append(AIMessage(ai_msg))

    print(f"AI: {ai_msg}")

print("------Chat History------")
print(chat_history)