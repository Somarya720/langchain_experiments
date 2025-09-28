from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=32)

doc_list = [
    'Messi won the world cup',
    'Gautam Gambhir is the coach', 
    'Trump is retarded', 
    'Haunted movies are the best'
]

vector_list = embeddings.embed_documents(doc_list)
print(vector_list)

for vector in vector_list:
    v = []
    for element in vector:
        v.append(round(element,2))

    print(v)