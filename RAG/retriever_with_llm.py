from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
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

query = "Is Harsh Neel a developer?"

document_list = retriever.invoke(query)

relevant_chunks = ''
for doc in document_list:
    relevant_chunks += f"Source: {doc.metadata['Source']}\nContent: {doc.page_content}"

print(relevant_chunks)