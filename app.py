import subprocess
import sys

# Function to install packages
def install(package):
    """Install the specified package using pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Updated list of required packages
required_packages = [
    "pinecone-client",
    "sentence-transformers",
    "streamlit",
    "scipy",
    "transformers",
    "numpy",
    "torch"
]

# Install necessary packages if not already installed
for package in required_packages:
    try:
        __import__(package.replace("-", "_"))
    except ImportError:
        install(package)

import os
import uuid
import re
import numpy as np
import torch
import pinecone
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st
from scipy.spatial.distance import cosine

def load_documents(folder_path):
    """
    Load text documents from a specified folder.

    Args:
        folder_path (str): Path to the folder containing text files.

    Returns:
        dict: A dictionary with filenames (without extension) as keys and file contents as values.
    """
    documents = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            filepath = os.path.join(folder_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    file_key = os.path.splitext(filename)[0]  # Strip extension
                    documents[file_key] = f.read()
            except IOError as e:
                st.error(f"Error reading {filepath}: {e}")
    return documents

def chunk_document(text, max_words=1000):
    """
    Split a text document into chunks of up to max_words words.

    Args:
        text (str): The text to be chunked.
        max_words (int): Maximum number of words per chunk.

    Returns:
        list: A list of text chunks.
    """
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = []
    word_count = 0
    for sentence in sentences:
        words_in_sentence = len(sentence.split())
        if word_count + words_in_sentence <= max_words:
            current_chunk.append(sentence)
            word_count += words_in_sentence
        else:
            chunks.append(' '.join(current_chunk).strip())
            current_chunk = [sentence]
            word_count = words_in_sentence
    if current_chunk:
        chunks.append(' '.join(current_chunk).strip())
    return chunks

def prepare_chunks(documents):
    """
    Prepare text chunks from documents for indexing.

    Args:
        documents (dict): Dictionary of documents.

    Returns:
        list: List of dictionaries containing chunk data.
    """
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

def index_chunks(chunks, model, index):
    """
    Generate embeddings for each chunk and upsert them into Pinecone.

    Args:
        chunks (list): List of chunk dictionaries with 'id', 'text', and 'metadata'.
        model (SentenceTransformer): The embedding model.
        index (pinecone.Index): The Pinecone index instance.
    """
    upsert_data = []
    for chunk in chunks:
        try:
            embedding = model.encode(chunk['text']).tolist()
            upsert_data.append((chunk['id'], embedding, chunk['metadata']))
        except Exception as e:
            st.error(f"Error encoding chunk {chunk['id']}: {e}")

    if upsert_data:
        index.upsert(vectors=upsert_data)

def generate_session_token():
    """Generate a unique session token."""
    return str(uuid.uuid4())

def display_introduction():
    """Display the introduction text in the Streamlit app."""
    introduction_text = """
    ## Welcome!

    Hi there! I'm thrilled to learn more about your brand. 😊

    **Objective:**  
    I'll be asking you a series of questions to gather detailed insights about your brand. Based on your responses, I'll classify your brand into primary and secondary archetypes. This classification will help us craft a marketing strategy that truly resonates with your brand's core identity and your audience.

    Let's dive in!
    """
    st.write(introduction_text)

def display_conclusion(primary, secondary):
    """
    Display the conclusion text with the classification results.

    Args:
        primary (str): The primary archetype.
        secondary (str): The secondary archetype (if any).
    """
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

def interview_flow(questions, responses_key='responses'):
    """
    Manage the interview flow by presenting questions and capturing responses.

    Args:
        questions (list): List of dictionaries with 'question' and 'follow_up'.
        responses_key (str): Key to store responses in session state.
    """
    if 'current_question' not in st.session_state:
        st.session_state['current_question'] = 0
        st.session_state['current_step'] = 'main'
        st.session_state[responses_key] = []
        display_introduction()

    current_q_idx = st.session_state['current_question']

    if current_q_idx >= len(questions):
        return

    current_step = st.session_state['current_step']
    current_question = questions[current_q_idx]

    if current_step == 'main':
        st.write(f"**Question {current_q_idx + 1}:** {current_question['question']}")
        main_response = st.text_area("Your Answer:", key=f"main_{current_q_idx}")

        if st.button("Next", key=f"next_main_{current_q_idx}"):
            if main_response.strip() == "":
                st.warning("Please provide an answer before proceeding.")
            else:
                st.session_state[responses_key].append({
                    'question': current_question['question'],
                    'response': main_response.strip()
                })
                st.session_state['current_step'] = 'follow_up'
                st.experimental_rerun()

    elif current_step == 'follow_up':
        st.write(f"**Follow-up Question:** {current_question['follow_up']}")
        follow_up_response = st.text_area("Your Answer:", key=f"follow_up_{current_q_idx}")

        if st.button("Next", key=f"next_follow_up_{current_q_idx}"):
            if follow_up_response.strip() == "":
                st.warning("Please provide an answer before proceeding.")
            else:
                st.session_state[responses_key][-1]['follow_up'] = follow_up_response.strip()
                st.session_state['current_step'] = 'main'
                st.session_state['current_question'] += 1
                st.experimental_rerun()

def get_archetype_embedding(archetype_name, index):
    """
    Retrieve and average embeddings for a given archetype from Pinecone.

    Args:
        archetype_name (str): The name of the archetype.
        index (pinecone.Index): The Pinecone index instance.

    Returns:
        list: Averaged embedding vector for the archetype.
    """
    try:
        query_result = index.query(
            filter={'archetype': {'$eq': archetype_name}},
            top_k=100,
            include_values=True
        )
    except Exception as e:
        st.error(f"Error querying Pinecone for archetype '{archetype_name}': {e}")
        return [0.0] * 384

    embeddings = [match['values'] for match in query_result['matches']]
    if not embeddings:
        return [0.0] * 384

    archetype_embedding = np.mean(embeddings, axis=0).tolist()
    return archetype_embedding

def analyze_sentiment(text, tokenizer, sentiment_model):
    """
    Analyze the sentiment of a given text using a pre-trained model.

    Args:
        text (str): The text to analyze.
        tokenizer (AutoTokenizer): The tokenizer for the sentiment model.
        sentiment_model (AutoModelForSequenceClassification): The sentiment analysis model.

    Returns:
        float: Sentiment score ranging from -1 (negative) to 1 (positive).
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    scores = outputs.logits.numpy()
    probabilities = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    sentiment_score = probabilities[0][1] - probabilities[0][0]
    return sentiment_score

def classify_archetypes(responses, documents, model, index, tokenizer, sentiment_model):
    """
    Classify the client into primary and secondary archetypes based on responses.

    Args:
        responses (list): List of response dictionaries.
        documents (dict): Dictionary of documents.
        model (SentenceTransformer): The embedding model.
        index (pinecone.Index): The Pinecone index instance.
        tokenizer (AutoTokenizer): The tokenizer for the sentiment model.
        sentiment_model (AutoModelForSequenceClassification): The sentiment analysis model.

    Returns:
        tuple: Primary archetype and secondary archetype (if any).
    """
    archetype_scores = {archetype: 0.0 for archetype in documents.keys()}

    for response in responses:
        response_text = response['response']
        if 'follow_up' in response:
            response_text += ' ' + response['follow_up']
        try:
            response_embedding = model.encode(response_text)
        except Exception as e:
            st.error(f"Error encoding response text: {e}")
            continue

        sentiment_score = analyze_sentiment(response_text, tokenizer, sentiment_model)

        for archetype in archetype_scores.keys():
            archetype_embedding = get_archetype_embedding(archetype, index)
            if archetype_embedding == [0.0] * len(archetype_embedding):
                continue
            similarity = 1 - cosine(response_embedding, archetype_embedding)
            if np.isnan(similarity):
                similarity = 0.0
            adjusted_score = similarity * (1 + sentiment_score)
            archetype_scores[archetype] += adjusted_score

    sorted_archetypes = sorted(archetype_scores.items(), key=lambda x: x[1], reverse=True)

    primary_archetype = sorted_archetypes[0][0]
    threshold = 0.9 * sorted_archetypes[0][1]

    secondary_archetype = (
        sorted_archetypes[1][0]
        if len(sorted_archetypes) > 1 and sorted_archetypes[1][1] > threshold
        else None
    )

    return primary_archetype, secondary_archetype

def main():
    """Main function to run the Streamlit app."""
    st.title("AI Chat Agent for Brand Archetype Classification")

    questions = [
        {
            'question': "What’s your primary goal in interacting with customers?",
            'follow_up': "Can you give a specific example where this goal was evident in a recent customer interaction?"
        },
        # ... (Add the rest of your questions here)
    ]

    interview_flow(questions)

    if st.session_state.get('current_question', 0) >= len(questions):
        st.write("## Processing Your Responses...")

        if 'processed' not in st.session_state:
            st.session_state['processed'] = True

            folder_path = 'knowledge_base'
            documents = load_documents(folder_path)
            if not documents:
                st.error("No documents found in the knowledge base.")
                st.stop()

            all_chunks = prepare_chunks(documents)

            openai_api_key = os.environ.get('OPENAI_API_KEY')
            pinecone_api_key = os.environ.get('PINECONE_API_KEY')
            pinecone_env = os.environ.get('PINECONE_ENVIRONMENT')

            if all([openai_api_key, pinecone_api_key, pinecone_env]):
                st.success("All API keys are set successfully!")
            else:
                st.error("Error: One or more API keys are missing.")
                st.stop()

            try:
                pinecone.init(
                    api_key=pinecone_api_key,
                    environment=pinecone_env
                )
            except Exception as e:
                st.error(f"Error initializing Pinecone: {e}")
                st.stop()

            try:
                existing_indexes = pinecone.list_indexes()
            except Exception as e:
                st.error(f"Error listing Pinecone indexes: {e}")
                st.stop()
            st.write(f"Existing Pinecone indexes: {existing_indexes}")

            index_name = 'knowledge-base'

            if index_name not in existing_indexes:
                try:
                    pinecone.create_index(
                        name=index_name,
                        dimension=384,
                        metric='cosine'
                    )
                    st.success(f"Created Pinecone index: {index_name}")
                except Exception as e:
                    st.error(f"Error creating Pinecone index '{index_name}': {e}")
                    st.stop()
            else:
                st.info(f"Pinecone index '{index_name}' already exists.")

            try:
                index = pinecone.Index(index_name)
            except Exception as e:
                st.error(f"Error connecting to Pinecone index '{index_name}': {e}")
                st.stop()

            try:
                model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                st.error(f"Error loading embedding model: {e}")
                st.stop()

            index_chunks(all_chunks, model, index)
            st.success("Indexed all chunks into Pinecone.")

            try:
                tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
                sentiment_model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
            except Exception as e:
                st.error(f"Error loading sentiment analysis model: {e}")
                st.stop()

            primary, secondary = classify_archetypes(
                st.session_state['responses'],
                documents,
                model,
                index,
                tokenizer,
                sentiment_model
            )

            st.write(f"**Primary Archetype:** {primary}")
            if secondary:
                st.write(f"**Secondary Archetype:** {secondary}")
            else:
                st.write("**No Secondary Archetype Detected.**")

            display_conclusion(primary, secondary)

    if st.button("Restart Interview"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()

if __name__ == "__main__":
    main()
