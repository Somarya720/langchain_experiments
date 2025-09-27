from dotenv import load_dotenv
import os

current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['HF_HOME'] = os.path.join(current_path, 'huggingface_cache')

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task = 'text-generation',
    pipeline_kwargs=dict(
        max_new_tokens = 100,
        temperature = 0.2
    )
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India?")

print(result)