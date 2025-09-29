from langchain_core.prompts.loading import load_prompt
from get_template_path import get_path

template_path = get_path('prompt.json')
template = load_prompt(template_path)
