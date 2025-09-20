from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import os
import sys

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

db = Chroma.from_documents(documents='', embedding='', persist_directory=persistant_directory)
