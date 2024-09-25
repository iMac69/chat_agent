import subprocess
import sys

# Function to install packages
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install necessary packages
required_packages = [
    "pinecone-client",
    "sentence-transformers",
    "streamlit",
    "scipy"
]

for package in required_packages:
    try:
        __import__(package.replace("-", "_"))
    except ImportError:
        install(package)

import os
import pinecone
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import streamlit as st
from scipy.spatial.distance import cosine

# Function to load documents from a folder
def load_documents(folder_path):
    documents = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                documents[filename] = f.read()
    return documents

# Function to chunk a document into smaller pieces
def chunk_document(text, max_words=1000):
    import re
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""
    word_count = 0
    for sentence in sentences:
        words_in_sentence = len(sentence.split())
        if word_count + words_in_sentence <= max_words:
            current_chunk += " " + sentence
            word_count += words_in_sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            word_count = words_in_sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Function to prepare chunks from documents
def prepare_chunks(documents):
    all_chunks = []
    for archetype_name, text in documents.items():
        chunks = chunk_document(text)
        for idx, chunk in enumerate(chunks):
            chunk_data = {
                'id': f"{archetype_name}_{idx}",
                'text': chunk,
                'metadata': {
                    'archetype': archetype_name,
                    'chunk_id': idx,
                    'total_chunks': len(chunks)
                }
            }
            all_chunks.append(chunk_data)
    return all_chunks

# Function to index chunks into Pinecone
def index_chunks(chunks, model, index):
    """
    Generate embeddings for each chunk and upsert them into Pinecone.
    
    Args:
        chunks (list): List of chunk dictionaries with 'id', 'text', and 'metadata'.
        model (SentenceTransformer): The embedding model.
        index (pinecone.Index): The Pinecone index instance.
    """
    # Prepare a list of tuples for upsert
    upsert_data = []
    for chunk in chunks:
        embedding = model.encode(chunk['text']).tolist()
        upsert_data.append((chunk['id'], embedding, chunk['metadata']))
    
    # Upsert all chunks in bulk for efficiency
    if upsert_data:
        index.upsert(vectors=upsert_data)

# Function to generate a session token (implementation needed)
def generate_session_token():
    import uuid
    return str(uuid.uuid4())

# Function to manage the interview flow using Streamlit
def interview_flow(questions, responses_key='responses'):
    """
    Manage the interview flow by presenting questions and capturing responses.
    
    Args:
        questions (list): List of dictionaries with 'question' and 'follow_up'.
        responses_key (str): Key to store responses in session state.
    """
    if 'session_token' not in st.session_state:
        st.session_state['session_token'] = generate_session_token()
    if responses_key not in st.session_state:
        st.session_state[responses_key] = []
    
    st.write("Hi there! I'm excited to learn more about your brand. 😊")
    
    for idx, q in enumerate(questions):
        with st.expander(f"Question {idx + 1}"):
            response = st.text_input(q['question'], key=f"q_{idx}")
            follow_up = st.text_input(q['follow_up'], key=f"q_{idx}_follow")
            if response and follow_up:
                st.session_state[responses_key].append({
                    'question': q['question'],
                    'follow_up': q['follow_up'],
                    'response': {
                        'answer': response,
                        'example': follow_up
                    }
                })

# Function to retrieve and average embeddings for a given archetype
def get_archetype_embedding(archetype_name, index):
    """
    Retrieve and average embeddings for a given archetype.
    
    Args:
        archetype_name (str): The name of the archetype.
        index (pinecone.Index): The Pinecone index instance.
        
    Returns:
        list: Averaged embedding vector for the archetype.
    """
    # Query Pinecone for all chunks related to the archetype
    query_result = index.query(
        filter={'archetype': archetype_name}, 
        top=100, 
        include_values=True
    )
    
    embeddings = [match['values'] for match in query_result['matches']]
    if not embeddings:
        return [0.0] * 384  # Return a zero vector if no embeddings found
    
    # Calculate the average embedding
    archetype_embedding = [sum(col) / len(col) for col in zip(*embeddings)]
    return archetype_embedding

# Function to classify archetypes based on responses
def classify_archetypes(responses, documents, model, index):
    """
    Classify the client into primary and secondary archetypes based on responses.
    
    Args:
        responses (list): List of response dictionaries.
        documents (dict): Dictionary of documents.
        model (SentenceTransformer): The embedding model.
        index (pinecone.Index): The Pinecone index instance.
        
    Returns:
        tuple: Primary archetype and secondary archetype (if any).
    """
    # Initialize a dictionary to hold cumulative similarity scores
    archetype_scores = {archetype: 0 for archetype in documents.keys()}
    
    for response in responses:
        response_text = response['response']['answer'] + ' ' + response['response']['example']
        response_embedding = model.encode(response_text)
        
        for archetype in archetype_scores.keys():
            archetype_embedding = get_archetype_embedding(archetype, index)
            score = 1 - cosine(response_embedding, archetype_embedding)
            archetype_scores[archetype] += score
    
    # Sort archetypes based on cumulative scores
    sorted_archetypes = sorted(archetype_scores.items(), key=lambda x: x[1], reverse=True)
    
    primary_archetype = sorted_archetypes[0][0]
    # Define threshold for secondary archetype (e.g., within 10% of primary score)
    threshold = 0.9 * sorted_archetypes[0][1]
    
    secondary_archetype = (
        sorted_archetypes[1][0] 
        if len(sorted_archetypes) > 1 and sorted_archetypes[1][1] > threshold 
        else None
    )
    
    return primary_archetype, secondary_archetype

# Main execution starts here
def main():
    st.title("AI Chat Agent for Brand Archetype Classification")
    
    # Example list of questions and follow-ups
    questions = [
        {
            'question': "What’s your primary goal in interacting with customers?",
            'follow_up': "Can you give a specific example?"
        },
        {
            'question': "How would you describe your ideal brand voice?",
            'follow_up': "Does it vary by platform or audience?"
        },
        # Add other questions as needed
    ]
    
    # Manage the interview flow
    interview_flow(questions)
    
    # When the user has completed all responses
    if st.button("Submit Responses"):
        if 'responses' in st.session_state and st.session_state['responses']:
            # Load documents
            folder_path = '/content/knowledge_base'  # Update this path as needed
            documents = load_documents(folder_path)
            
            # Prepare chunks
            all_chunks = prepare_chunks(documents)
            
            # Initialize and Configure Pinecone
            openai_api_key = os.environ.get('OPENAI_API_KEY')      # Ensure these are set
            pinecone_api_key = os.environ.get('PINECONE_API_KEY')
            pinecone_env = os.environ.get('PINECONE_ENVIRONMENT')
            
            # Verify that the API keys are set
            if all([openai_api_key, pinecone_api_key, pinecone_env]):
                st.success("All API keys are set successfully!")
            else:
                st.error("Error: One or more API keys are missing.")
                return
            
            # Initialize Pinecone
            pc = Pinecone(
                api_key=pinecone_api_key,    # Your Pinecone API key
                environment=pinecone_env      # Your Pinecone environment (e.g., 'us-east1-gcp')
            )
            
            # Check existing indexes
            existing_indexes = pc.list_indexes().names()  # Correctly call the 'names' method
            st.write(f"Existing Pinecone indexes: {existing_indexes}")
            
            # Define your index name
            index_name = 'knowledge-base'
            
            # Create a new index if it doesn't exist
            if index_name not in existing_indexes:
                pc.create_index(
                    name=index_name,
                    dimension=384,  # embedding size of the all-MiniLM-L6-v2 model
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',        # Choose your cloud provider ('aws', 'gcp', etc.)
                        region='us-east-1'   # Choose the appropriate region
                    )
                )
                st.success(f"Created Pinecone index: {index_name}")
            else:
                st.info(f"Pinecone index '{index_name}' already exists.")
            
            # Connect to the index
            index = pc.Index(index_name)
            
            # Load the embedding model
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Index the chunks
            index_chunks(all_chunks, model, index)
            st.success("Indexed all chunks into Pinecone.")
            
            # Classify archetypes based on responses
            primary, secondary = classify_archetypes(
                st.session_state['responses'], 
                documents, 
                model, 
                index
            )
            
            st.write(f"**Primary Archetype:** {primary}")
            if secondary:
                st.write(f"**Secondary Archetype:** {secondary}")
            else:
                st.write("**No Secondary Archetype Detected.**")
        else:
            st.error("Please complete all responses before submitting.")

if __name__ == "__main__":
    main()
