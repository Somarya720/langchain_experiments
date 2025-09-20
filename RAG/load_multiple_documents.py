from dotenv import load_dotenv
from langchain_chroma import Chroma
import os
import sys

# load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
persistant_dir = os.path.join(current_dir, 'db', 'chroma_multiple')
doc_path = os.path.join(current_dir, 'documents')

if os.path.exists(persistant_dir):
    print("Vectore store already created")
    sys.exit(0)

doc_list = [doc for doc in os.listdir(doc_path) if doc.endswith('.txt')]

print(doc_list)
