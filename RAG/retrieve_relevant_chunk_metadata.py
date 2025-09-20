from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embedding = OpenAIEmbeddings(
    model='text-embedding-3-small'
)

current_dir = os.path.dirname(os.path.abspath(__file__))
persist_dir = os.path.join(current_dir, 'db', 'chroma_multiple')

# initialize chroma client
db = Chroma(embedding_function=embedding, persist_directory=persist_dir)

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.2}
)

query = "Where was dracula's castle?"
document_list = retriever.invoke(query)

for document in document_list:
    print(document.page_content, end='\n\n\n')
    print(document.metadata)