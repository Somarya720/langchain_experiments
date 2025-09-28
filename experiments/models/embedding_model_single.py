from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=32)

text = "Messi won the world cup in 2022"

vector = embeddings.embed_query(text=text)

print(vector)

