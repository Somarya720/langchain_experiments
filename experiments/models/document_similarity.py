from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers.",
    "Spiderman was played by Tobey Maguire",
    "Superman the man of steel was released on 2013"
]

query = 'tell me about bumrah'

embeddings = OpenAIEmbeddings(model = 'text-embedding-3-small', dimensions=512)

# create embeddings
doc_vectors = embeddings.embed_documents(documents)
query_vector = embeddings.embed_query(query)

# specify threshold
threshold = 0.3

# calculate similarity score
similarity_matrix = cosine_similarity([query_vector], doc_vectors)
doc_score = list(zip(documents, similarity_matrix[0]))

# filter documents over threshold
filtered_doc_score = [doc_sc for doc_sc in doc_score if doc_sc[1] > threshold]
print(filtered_doc_score)

if filtered_doc_score:
    # get max score
    max_element = max(filtered_doc_score, key=lambda x: x[1])
    print(max_element[0])
else:
    print("None of the documents match to the query")
