import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
vector_db_dir = os.path.join(current_dir,'db','chroma')

embeddings = OpenAIEmbeddings(
    model='text-embedding-3-small'
)

# initial chroma client
db = Chroma(persist_directory=vector_db_dir, embedding_function=embeddings)

# initialize retriever
retriever = db.as_retriever(
    search_type = 'similarity_score_threshold',
    search_kwargs = {
        'k': 3,
        'score_threshold': 0.5
    }
)

query = 'Where does Gandalf meet Frodo?'

doc_list = retriever.invoke(query)

for doc in doc_list:
    print('------')
    print(doc.page_content)