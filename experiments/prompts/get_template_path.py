import os

def get_path(file_name: str):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(current_dir, file_name)
    return template_path