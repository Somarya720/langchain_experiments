from langchain_huggingface.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

text = 'This is a text document'

vector = embeddings.embed_query(text)

print(vector[:5])

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

vector_list = embeddings.embed_documents(documents)

for vector in vector_list:
    print(vector[:5])