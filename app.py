import os
import streamlit as st
from sentence_transformers import SentenceTransformer
import pinecone
from sklearn.metrics.pairwise import cosine_similarity
import uuid
from typing import List, Dict, Tuple
from scipy.spatial.distance import cosine

# --------------------------- Configuration --------------------------- #

# Load API keys from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')  # e.g., 'us-east1-gcp'

# Constants
INDEX_NAME = 'knowledge-base'
EMBEDDING_DIMENSION = 384  # for 'all-MiniLM-L6-v2' model
CHUNK_SIZE = 1000  # Number of characters per chunk
CHUNK_WORD_LIMIT = 200  # Approximate word limit per chunk

# --------------------------- Helper Functions --------------------------- #

def load_documents(folder_path: str) -> Dict[str, str]:
    """
    Load documents from the specified folder. Each file represents an archetype.

    Args:
        folder_path (str): Path to the folder containing document files.

    Returns:
        Dict[str, str]: A dictionary mapping archetype names to their text content.
    """
    documents = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            archetype_name = os.path.splitext(filename)[0]
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                documents[archetype_name] = file.read()
    return documents

def chunk_document(text: str) -> List[str]:
    """
    Split the text into chunks based on word count.

    Args:
        text (str): The text to be chunked.

    Returns:
        List[str]: A list of text chunks.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_word_count = 0

    for word in words:
        current_chunk.append(word)
        current_word_count += 1
        if current_word_count >= CHUNK_WORD_LIMIT:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_word_count = 0

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def prepare_chunks(documents: Dict[str, str]) -> List[Dict]:
    """
    Prepare document chunks with metadata for indexing.

    Args:
        documents (Dict[str, str]): Mapping of archetype names to text.

    Returns:
        List[Dict]: List of chunk data dictionaries.
    """
    all_chunks = []
    for archetype_name, text in documents.items():
        chunks = chunk_document(text)
        for idx, chunk in enumerate(chunks):
            chunk_data = {
                'id': f"{archetype_name}_{idx}_{uuid.uuid4()}",
                'text': chunk,
                'metadata': {
                    'archetype': archetype_name,
                    'chunk_id': idx,
                    'total_chunks': len(chunks)
                }
            }
            all_chunks.append(chunk_data)
    return all_chunks

def initialize_pinecone(api_key: str, environment: str) -> pinecone.Index:
    """
    Initialize Pinecone and return the index.

    Args:
        api_key (str): Pinecone API key.
        environment (str): Pinecone environment.

    Returns:
        pinecone.Index: The Pinecone index object.
    """
    pinecone.init(api_key=api_key, environment=environment)
    if INDEX_NAME not in pinecone.list_indexes():
        pinecone.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric='cosine'
        )
        st.success(f"Pinecone index '{INDEX_NAME}' created.")
    else:
        st.info(f"Pinecone index '{INDEX_NAME}' already exists.")
    return pinecone.Index(INDEX_NAME)

def index_chunks(index: pinecone.Index, chunks: List[Dict], model: SentenceTransformer):
    """
    Index chunks into Pinecone.

    Args:
        index (pinecone.Index): The Pinecone index.
        chunks (List[Dict]): List of chunk data dictionaries.
        model (SentenceTransformer): The embedding model.
    """
    upsert_data = []
    for chunk in chunks:
        embedding = model.encode(chunk['text']).tolist()
        upsert_data.append((chunk['id'], embedding, chunk['metadata']))

    if upsert_data:
        index.upsert(vectors=upsert_data)
        st.success(f"Upserted {len(upsert_data)} vectors into Pinecone.")
    else:
        st.warning("No data to upsert into Pinecone.")

def get_archetype_embedding(index: pinecone.Index, archetype_name: str) -> List[float]:
    """
    Retrieve and average embeddings for a given archetype.

    Args:
        index (pinecone.Index): The Pinecone index.
        archetype_name (str): The name of the archetype.

    Returns:
        List[float]: Averaged embedding vector for the archetype.
    """
    query_result = index.query(
        filter={'archetype': archetype_name},
        top=100,
        include_values=True
    )

    embeddings = [match['values'] for match in query_result['matches']]
    if not embeddings:
        return [0.0] * EMBEDDING_DIMENSION  # Return a zero vector if no embeddings found

    # Calculate the average embedding
    averaged_embedding = [sum(col) / len(col) for col in zip(*embeddings)]
    return averaged_embedding

def classify_archetypes(index: pinecone.Index, documents: Dict[str, str], responses: List[Dict], model: SentenceTransformer) -> Tuple[str, str]:
    """
    Classify the client into primary and secondary archetypes based on responses.

    Args:
        index (pinecone.Index): The Pinecone index.
        documents (Dict[str, str]): Mapping of archetype names to text.
        responses (List[Dict]): List of user responses.
        model (SentenceTransformer): The embedding model.

    Returns:
        Tuple[str, str]: Primary archetype and secondary archetype (if any).
    """
    archetype_scores = {archetype: 0 for archetype in documents.keys()}

    for response in responses:
        response_text = response['response']['answer'] + ' ' + response['response']['example']
        response_embedding = model.encode(response_text)

        for archetype in archetype_scores.keys():
            archetype_embedding = get_archetype_embedding(index, archetype)
            if all(e == 0.0 for e in archetype_embedding):
                continue  # Skip if archetype embedding is a zero vector
            score = 1 - cosine(response_embedding, archetype_embedding)
            archetype_scores[archetype] += score

    # Sort archetypes based on cumulative scores
    sorted_archetypes = sorted(archetype_scores.items(), key=lambda x: x[1], reverse=True)

    primary_archetype = sorted_archetypes[0][0]
    threshold = 0.9 * sorted_archetypes[0][1]
    secondary_archetype = sorted_archetypes[1][0] if len(sorted_archetypes) > 1 and sorted_archetypes[1][1] > threshold else None

    return primary_archetype, secondary_archetype

# --------------------------- Streamlit App --------------------------- #

def main():
    st.set_page_config(page_title="AI Chat Agent", layout="wide")
    st.title("AI Chat Agent for Archetype Classification")

    # Verify API keys
    if not all([OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT]):
        st.error("Error: One or more API keys are missing. Please set OPENAI_API_KEY, PINECONE_API_KEY, and PINECONE_ENVIRONMENT as environment variables.")
        return

    # Initialize Pinecone
    index = initialize_pinecone(PINECONE_API_KEY, PINECONE_ENVIRONMENT)

    # Load and prepare documents
    folder_path = 'knowledge_base'  # Ensure this path is correct
    if not os.path.exists(folder_path):
        st.error(f"Knowledge base folder '{folder_path}' not found. Please ensure the folder exists and contains archetype `.txt` files.")
        return

    documents = load_documents(folder_path)
    all_chunks = prepare_chunks(documents)

    # Load embedding model
    with st.spinner("Loading embedding model..."):
        model = SentenceTransformer('all-MiniLM-L6-v2')

    # Index chunks if not already indexed
    if st.button("Index Documents into Pinecone"):
        index_chunks(index, all_chunks, model)

    st.markdown("---")

    # Define the list of questions and follow-ups
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

    def interview_flow():
        """
        Manage the interview flow by presenting questions and capturing responses.
        """
        if 'responses' not in st.session_state:
            st.session_state['responses'] = []

        st.write("Hi there! I'm excited to learn more about your brand. 😊")

        for idx, q in enumerate(questions):
            with st.expander(f"Question {idx + 1}"):
                response = st.text_input(q['question'], key=f"q_{idx}")
                follow_up = st.text_input(q['follow_up'], key=f"q_{idx}_follow")
                if st.button(f"Submit Response {idx + 1}"):
                    if response and follow_up:
                        st.session_state['responses'].append({
                            'question': q['question'],
                            'follow_up': q['follow_up'],
                            'response': {
                                'answer': response,
                                'example': follow_up
                            }
                        })
                        st.success("Response submitted!")
                    else:
                        st.warning("Please provide both an answer and an example.")

        if st.session_state['responses']:
            if st.button("Classify Archetypes"):
                with st.spinner("Classifying archetypes..."):
                    primary, secondary = classify_archetypes(index, documents, st.session_state['responses'], model)
                    st.subheader("Classification Results")
                    st.write(f"**Primary Archetype:** {primary}")
                    if secondary:
                        st.write(f"**Secondary Archetype:** {secondary}")
                    else:
                        st.write("**Secondary Archetype:** None")

    interview_flow()

if __name__ == "__main__":
    main()
