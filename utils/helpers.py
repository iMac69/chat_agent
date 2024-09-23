# utils/helpers.py (Updated)

import os
import uuid
import json
import logging
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env
load_dotenv()

# Configure logging
logging.basicConfig(
    filename='chat_agent.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def generate_session_token():
    """
    Generate a unique session token using UUID4.
    """
    return str(uuid.uuid4())

def load_knowledge_base():
    """
    Load knowledge base documents from the knowledge_base directory.
    Returns a dictionary with document names as keys and content as values.
    """
    knowledge_dir = os.path.join(os.getcwd(), 'knowledge_base')
    knowledge = {}
    for filename in os.listdir(knowledge_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(knowledge_dir, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                key = filename.replace('.txt', '')
                knowledge[key] = content
    return knowledge

def save_json(data, filename):
    """
    Save a dictionary as a JSON file.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_json(filename):
    """
    Load a JSON file and return its content as a dictionary.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)