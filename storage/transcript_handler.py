# storage/transcript_handler.py (Updated)

import json
import os
import gspread
from google.oauth2.service_account import Credentials
from utils.helpers import save_json, load_json, logging
import pandas as pd

# Load environment variables
GOOGLE_SHEETS_CREDENTIALS = os.getenv('GOOGLE_SHEETS_CREDENTIALS')
GOOGLE_SHEETS_SPREADSHEET_NAME = os.getenv('GOOGLE_SHEETS_SPREADSHEET_NAME')

# Authenticate and initialize Google Sheets client
def init_gspread():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    credentials = Credentials.from_service_account_file(GOOGLE_SHEETS_CREDENTIALS, scopes=scopes)
    client = gspread.authorize(credentials)
    return client

def export_to_google_sheets(transcript_json):
    """
    Export the conversation transcript to Google Sheets.
    
    :param transcript_json: Dictionary containing the transcript data.
    """
    client = init_gspread()
    try:
        sheet = client.open(GOOGLE_SHEETS_SPREADSHEET_NAME).sheet1
    except Exception as e:
        logging.error(f"Error opening Google Sheet: {e}")
        return
    
    # Prepare data for export
    client_id = transcript_json.get('client_id', 'N/A')
    timestamp = transcript_json.get('timestamp', 'N/A')
    archetypes = transcript_json.get('archetypes', {})
    primary = archetypes.get('primary', 'N/A')
    secondary = archetypes.get('secondary', 'N/A')
    
    # Prepare questions and responses
    interview = transcript_json.get('interview', [])
    data = {
        'Client ID': client_id,
        'Timestamp': timestamp,
        'Primary Archetype': primary,
        'Secondary Archetype': secondary
    }
    
    for idx, qa in enumerate(interview, start=1):
        question = qa.get('question', '')
        response = qa.get('response', {}).get('answer', '')
        follow_up = qa.get('follow_up', '')
        example = qa.get('response', {}).get('example', '')
        
        data[f'Question {idx}'] = question
        data[f'Response {idx}'] = response
        data[f'Follow-Up {idx}'] = follow_up
        data[f'Example {idx}'] = example
    
    # Convert to DataFrame
    df = pd.DataFrame([data])
    
    # Append to Google Sheet
    try:
        sheet.append_rows(df.values.tolist())
        logging.info(f"Transcript exported to Google Sheets for Client ID: {client_id}")
    except Exception as e:
        logging.error(f"Error exporting to Google Sheets: {e}")