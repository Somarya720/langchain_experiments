from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import os
import sys

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
persistant_dir = os.path.join(current_dir, 'db', 'chroma_multiple')
doc_path = os.path.join(current_dir, 'documents')

if os.path.exists(persistant_dir):
    print("Vectore store already created")
    sys.exit(0)

doc_list = [doc for doc in os.listdir(doc_path) if doc.endswith('.txt')]

if not doc_list:
    raise FileNotFoundError("No txt document found in the folder")

embedding = OpenAIEmbeddings(
    model='text-embedding-3-small'
)

documents = []

for doc in doc_list:

    loader = TextLoader(os.path.join(doc_path, doc), encoding='utf-8')
    document = loader.load()

    # Add source as the metadata
    document[0].metadata = {
        'Source': doc 
    }
    documents.append(document[0])
    
# Split text
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
document_chunks = splitter.split_documents(documents)

# create store
db = Chroma.from_documents(embedding=embedding, documents=document_chunks, persist_directory=persistant_dir)
