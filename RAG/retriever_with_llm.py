from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

current_path = os.path.dirname(os.path.abspath(__file__))
persist_dir = os.path.join(current_path, 'db', 'chroma_multiple')

embedding = OpenAIEmbeddings(
    model='text-embedding-3-small'
)

llm = ChatOpenAI(model='gpt-4.1-nano')

db = Chroma(embedding_function=embedding, persist_directory=persist_dir)

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.2}
)

query = "Where was dracula's castle located?"

document_list = retriever.invoke(query)

relevant_chunks = 'Relevant Documents: \n'
for doc in document_list:
    relevant_chunks += f"Source: {doc.metadata['Source']}\nContent: {doc.page_content}"

prompt = relevant_chunks + '\n\n' + query #+ "Please provide the answer based on the document provided earlier. Also mention the source. If the information is not in the documents then return this information does not exist in the document and dont answer general knowledge questions."

message_list = [
    SystemMessage("You are a helpful AI Assistant. Pleawse provide the answer based on the relevant document provided. Also mention the source. If the information is not in the document then return this information does not exist in the document. Dont answer general knowledge questions."),
    HumanMessage(prompt)
]
result = llm.invoke(message_list)

print(result.content)