import subprocess
import sys

# Function to install packages
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install necessary packages if not already installed
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

# Function to display the introduction
def display_introduction():
    introduction_text = """
    ## Welcome!
    
    Hi there! I'm thrilled to learn more about your brand. 😊
    
    **Objective:**  
    I'll be asking you a series of questions to gather detailed insights about your brand. Based on your responses, I'll classify your brand into primary and secondary archetypes. This classification will help us craft a marketing strategy that truly resonates with your brand's core identity and your audience.
    
    Let's dive in!
    """
    st.write(introduction_text)

# Function to display the conclusion
def display_conclusion(primary, secondary):
    conclusion_text = f"""
    ## Thank You!
    
    I truly appreciate you taking the time to share your insights. Based on your responses, your brand aligns with the following archetypes:
    
    **Primary Archetype:** {primary}
    """
    if secondary:
        conclusion_text += f"\n**Secondary Archetype:** {secondary}"
    conclusion_text += """
    
    This classification will guide us in crafting a tailored marketing strategy that resonates deeply with your audience and aligns perfectly with your brand's identity.
    
    If you have any feedback or would like to discuss these archetypes further, feel free to let me know!
    """
    st.write(conclusion_text)

# Function to manage the interview flow using Streamlit
def interview_flow(questions, responses_key='responses'):
    """
    Manage the interview flow by presenting questions and capturing responses in a conversational manner.
    
    Args:
        questions (list): List of dictionaries with 'question' and 'follow_up'.
        responses_key (str): Key to store responses in session state.
    """
    # Initialize session state variables if not present
    if 'current_question' not in st.session_state:
        st.session_state['current_question'] = 0  # Tracks the current question index
        st.session_state['current_step'] = 'main'  # Tracks whether to show 'main' question or 'follow_up'
        st.session_state[responses_key] = []       # Stores the responses
    
    current_q_idx = st.session_state['current_question']
    
    # If all questions have been answered, do not display more questions
    if current_q_idx >= len(questions):
        return
    
    current_step = st.session_state['current_step']
    current_question = questions[current_q_idx]
    
    if current_step == 'main':
        # Display the main question
        st.write(current_question['question'])
        main_response = st.text_input("Your Answer:", key=f"main_{current_q_idx}")
        
        if st.button("Next", key=f"next_main_{current_q_idx}"):
            if main_response.strip() == "":
                st.warning("Please provide an answer before proceeding.")
            else:
                # Save the main response
                st.session_state[responses_key].append({
                    'question': current_question['question'],
                    'response': main_response.strip()
                })
                # Move to follow-up step
                st.session_state['current_step'] = 'follow_up'
                st.experimental_rerun()
    
    elif current_step == 'follow_up':
        # Display the follow-up question
        st.write(current_question['follow_up'])
        follow_up_response = st.text_input("Your Answer:", key=f"follow_up_{current_q_idx}")
        
        if st.button("Next", key=f"next_follow_up_{current_q_idx}"):
            if follow_up_response.strip() == "":
                st.warning("Please provide an answer before proceeding.")
            else:
                # Save the follow-up response
                st.session_state[responses_key][-1]['follow_up'] = follow_up_response.strip()
                # Reset step and move to next question
                st.session_state['current_step'] = 'main'
                st.session_state['current_question'] += 1
                # Clear input fields and rerun
                st.experimental_rerun()

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
        response_text = response['response']
        if 'follow_up' in response:
            response_text += ' ' + response['follow_up']
        response_embedding = model.encode(response_text)
        
        for archetype in archetype_scores.keys():
            archetype_embedding = get_archetype_embedding(archetype, index)
            score = 1 - cosine(response_embedding, archetype_embedding)
            archetype_scores[archetype] += score
    
    # Sort archetypes based on cumulative scores
    sorted_archetypes = sorted(archetype_scores.items(), key=lambda x: x[1], reverse=True)
    
    primary_archetype = sorted_archetypes[0][0]
    # Define threshold for secondary archetype (e.g., within 90% of primary score)
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
    
    # Define the list of questions and follow-ups
    questions = [
        {
            'question': "What’s your primary goal in interacting with customers?",
            'follow_up': "Can you give a specific example where this goal was evident in a recent customer interaction?"
        },
        {
            'question': "How would you describe your ideal brand voice?",
            'follow_up': "Does your brand voice differ depending on the platform or audience, or is it consistent across all interactions?"
        },
        {
            'question': "What values are most important to your brand?",
            'follow_up': "Can you provide a real-world example where your brand has embodied these values?"
        },
        {
            'question': "How do you want customers to feel when interacting with your brand?",
            'follow_up': "Can you describe a customer testimonial or feedback that demonstrates this feeling?"
        },
        {
            'question': "What kind of imagery resonates most with your brand identity?",
            'follow_up': "How do you typically use this imagery in your marketing materials or online presence?"
        },
        {
            'question': "How does your brand approach innovation and change?",
            'follow_up': "Can you give an example of a recent innovation or change within your company?"
        },
        {
            'question': "What role does tradition play in your brand’s identity?",
            'follow_up': "How does your brand communicate this balance between tradition and innovation?"
        },
        {
            'question': "How does your brand handle adversity or setbacks?",
            'follow_up': "Can you describe a recent challenge your brand faced and how it was addressed?"
        },
        {
            'question': "What type of story does your brand most want to tell?",
            'follow_up': "Is there a specific campaign or marketing effort that captured this story well?"
        },
        {
            'question': "How would your brand approach the concept of luxury and indulgence?",
            'follow_up': "Does your brand differentiate between luxury and everyday offerings? How?"
        },
        {
            'question': "What role does aesthetic beauty play in your brand’s identity?",
            'follow_up': "Can you give an example of how aesthetic beauty is reflected in your products or services?"
        },
        {
            'question': "How does your brand approach moments of celebration and joy?",
            'follow_up': "Has your brand recently celebrated a milestone or event? How did you share that celebration with your audience?"
        },
        {
            'question': "How does your brand view success and achievement?",
            'follow_up': "Can you share an example of a major achievement for your brand?"
        },
        {
            'question': "What feeling do you think most motivates your customers to take a desired action?",
            'follow_up': "Can you share a customer success story where these feelings led to action?"
        },
        {
            'question': "How does your brand handle the unknown and uncertainty?",
            'follow_up': "Can you describe a time when your brand navigated uncertainty and how it maintained its values?"
        }
    ]
    
    # Manage the interview flow
    interview_flow(questions)
    
    # After all questions have been answered
    if st.session_state['current_question'] >= len(questions):
        st.write("## Processing Your Responses...")
        
        # Prevent re-processing
        if 'processed' not in st.session_state:
            st.session_state['processed'] = True
            
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
                st.stop()
            
            # Initialize Pinecone
            pc = Pinecone(
                api_key=pinecone_api_key,    # Your Pinecone API key
                environment=pinecone_env      # Your Pinecone environment (e.g., 'us-east1-gcp')
            )
            
            # Check existing indexes
            existing_indexes = pc.list_indexes().names()  # Correctly call the 'names' method
            st.write(f"Existing Pinecone indexes: {existing_indexes}")
            
            # Define your index name
           ​⬤
