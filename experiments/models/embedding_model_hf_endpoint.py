from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings()

text = 'this is a sample text'
query_result = embeddings.embed_query(text)
print(query_result[:3])