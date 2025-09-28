from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = 'tell me about bumrah'

embeddings = OpenAIEmbeddings(model = 'text-embedding-3-small', dimensions=100)

doc_vectors = embeddings.embed_documents(documents)
query_vector = embeddings.embed_query(query)

similarity_matrix = cosine_similarity([query_vector], doc_vectors)

doc_score = list(zip(documents, similarity_matrix[0]))
max_element = max(doc_score, key=lambda x: x[1])

print(max_element[0])
