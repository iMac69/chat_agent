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
    "numpy"
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
import pinecone
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import streamlit as st
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the sentiment analysis model
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
sentiment_model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

def analyze_sentiment(text):
    """
    Analyze the sentiment of the given text.

    Args:
        text (str): The input text.

    Returns:
        float: Sentiment score ranging from -1 (negative) to 1 (positive).
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True)
    outputs = sentiment_model(**inputs)
    scores = outputs.logits.detach().numpy()
    # Softmax to get probabilities
    probabilities = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    sentiment_score = probabilities[0][1] - probabilities[0][0]  # Positive minus negative
    return sentiment_score

def load_documents(folder_path):
    """
    Load text documents from the specified folder.

    Args:
        folder_path (str): Path to the folder containing text files.

    Returns:
        dict: A dictionary mapping filenames to their content.
    """
    documents = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                documents[filename] = f.read()
    return documents

def chunk_document(text, max_words=1000):
    """
    Split a document into chunks of approximately max_words.

    Args:
        text (str): The input text.
        max_words (int): Maximum number of words per chunk.

    Returns:
        list: A list of text chunks.
    """
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

def prepare_chunks(documents):
    """
    Prepare chunks from the provided documents.

    Args:
        documents (dict): A dictionary of documents.

    Returns:
        list: A list of chunk dictionaries with metadata.
    """
    all_chunks = []
    for archetype_name, text in documents.items():
        chunks = chunk_document(text)
        for idx, chunk in enumerate(chunks):
            chunk_data = {
                'id': f"{archetype_name}_{idx}",
                'text': chunk,
                'metadata': {
                    'archetype': archetype_name.replace('.txt', ''),
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
    # Prepare a list of tuples for upsert
    upsert_data = []
    for chunk in chunks:
        embedding = model.encode(chunk['text']).tolist()
        upsert_data.append((chunk['id'], embedding, chunk['metadata']))

    # Upsert all chunks in bulk for efficiency
    if upsert_data:
        index.upsert(vectors=upsert_data)

def generate_session_token():
    """Generate a unique session token."""
    return str(uuid.uuid4())

def display_introduction():
    """Display the introduction text."""
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
    Display the conclusion with the classified archetypes.

    Args:
        primary (str): Primary archetype.
        secondary (str): Secondary archetype.
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
                # Rerun the app to update the UI
                st.experimental_rerun()

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
    # Using a dummy vector and filter to retrieve all vectors with the specified archetype
    query_result = index.query(
        vector=[0.0] * 384,  # Dummy vector since we're filtering by metadata
        filter={'archetype': archetype_name},
        top_k=1000,  # Adjust based on expected number of chunks per archetype
        include_values=True
    )

    embeddings = [match['values'] for match in query_result['matches']]
    if not embeddings:
        return [0.0] * 384  # Return a zero vector if no embeddings found

    # Calculate the average embedding
    archetype_embedding = np.mean(embeddings, axis=0).tolist()
    return archetype_embedding

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
    archetype_scores = {archetype.replace('.txt', ''): 0 for archetype in documents.keys()}

    for response in responses:
        response_text = response['response']
        if 'follow_up' in response:
            response_text += ' ' + response['follow_up']
        response_embedding = model.encode(response_text)

        # Analyze sentiment of the response
        sentiment_score = analyze_sentiment(response_text)

        for archetype in archetype_scores.keys():
            archetype_embedding = get_archetype_embedding(archetype, index)
            similarity_score = 1 - cosine(response_embedding, archetype_embedding)
            # Handle cases where cosine similarity might return NaN
            if np.isnan(similarity_score):
                similarity_score = 0
            # Adjust the similarity score by sentiment_score
            adjusted_score = similarity_score * (1 + sentiment_score)
            archetype_scores[archetype] += adjusted_score

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

def main():
    """Main function to run the Streamlit app."""
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

    # Display introduction if not already done
    if 'introduction_displayed' not in st.session_state:
        display_introduction()
        st.session_state['introduction_displayed'] = True

    # Manage the interview flow
    interview_flow(questions)

    # After all questions have been answered
    if st.session_state['current_question'] >= len(questions):
        st.write("## Processing Your Responses...")

        # Prevent re-processing
        if 'processed' not in st.session_state:
            st.session_state['processed'] = True

            # Load documents
            folder_path = 'knowledge_base'  # Update this path as needed
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

            # Initialize Pinecone (leave as is)
            pc = Pinecone(
                api_key=pinecone_api_key,
                environment=pinecone_env
            )

            # Check existing indexes
            existing_indexes = pc.list_indexes().names  # Correctly call the 'names' attribute
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

            # Display classification results
            st.write(f"**Primary Archetype:** {primary}")
            if secondary:
                st.write(f"**Secondary Archetype:** {secondary}")
            else:
                st.write("**No Secondary Archetype Detected.**")

            # Display Conclusion
            display_conclusion(primary, secondary)

    # Reset functionality (optional)
    if st.button("Restart Interview"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()

if __name__ == "__main__":
    main()
