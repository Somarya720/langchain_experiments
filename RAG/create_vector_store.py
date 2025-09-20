from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# get required directories
doc_name = 'lord_of_the_rings.txt'
current_path = os.path.dirname(os.path.abspath(__file__))
persistant_directory = os.path.join(current_path, 'db', 'chroma')
doc_path = os.path.join(current_path, 'documents', doc_name)

if os.path.exists(persistant_directory):
    print('Persistant directory exists no need to create chroma db')
    sys.exit(0)

if not os.path.exists(doc_path):
    raise FileNotFoundError(f"Doc Name {doc_name} does not exist")

# Load document 
loader = TextLoader(doc_path)
document = loader.load()

# Split into chunks of 1000 size each with overlap of 50
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
document_list = text_splitter.split_documents(document)

print(document_list[0].page_content)
print(f"number of chunks: {len(document_list)}")

embedding = OpenAIEmbeddings(
    model='text-embedding-3-small'
)

db = Chroma.from_documents(documents=document_list, embedding=embedding, persist_directory=persistant_directory)
