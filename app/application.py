# A simple flask application that 
# 1. reads the db file into a df
# 2. performs cleanup and etl
# 3. serves the df as a json endpoint
# 4. serves a simple html page to view the data
# 5. the frontend should have a simple and modern bootstrap UI design that will mimic functionality of the demo.ipynb notebook
# 6. simple search for filtering, and a view that shows the filtered text of the pdf

from flask import Flask, jsonify, render_template, request
import pandas as pd
import sqlite3
from bs4 import BeautifulSoup
import boto3
import os
import json
from dotenv import load_dotenv
from collections import Counter
import re

application = Flask(__name__)
load_dotenv()

def load_and_clean_db(db_name='pdf_texts.db'):
    conn = sqlite3.connect(db_name)
    df = pd.read_sql_query("SELECT * FROM pdf_texts", conn)
    conn.close()
    # ETL: parse html_metadata into columns
    def parse_html_metadata(html):
        soup = BeautifulSoup(html, 'html.parser')
        data = {}
        for div in soup.find_all('div', class_='flex-col'):
            label = div.find('strong')
            value = div.find('span')
            if label and value:
                key = label.text.strip().replace(':', '')
                val = value.text.strip()
                data[key] = val
        return data
    parsed = df['html_metadata'].apply(parse_html_metadata)
    parsed_df = parsed.apply(pd.Series)
    df = pd.concat([df, parsed_df], axis=1)
    return df

df = load_and_clean_db()

@application.route('/data.json')
def data_json():
    # Return the full DataFrame as JSON
    return df.to_json(orient='records')

# give me a route that shows the df to_html with bootstrap styling
@application.route('/data.html')
def data_html():
    # Convert DataFrame to HTML with Bootstrap styling
    # show only one line of text per row
    df
    # Create a copy of the DataFrame with truncated text
    display_df = df.copy()
    display_df['text'] = display_df['text'].str[1000:1100] + '...'
    html_table = display_df.to_html(classes='table table-striped table-bordered', index=False)
    return render_template('data.html', table=html_table)

@application.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '')
    if not query:
        return jsonify({'error': 'No search query provided'}), 400
    
    # Perform a case-insensitive search in the 'text' column
    results = df[df['text'].str.contains(query, case=False, na=False)].copy()
    
    if results.empty:
        return jsonify({'message': 'No results found'}), 404
    
    # Add a 'snippet' column with ±50 chars around the first search hit
    def make_snippet(text):
        text_lower = text.lower()
        idx = text_lower.find(query.lower())
        if idx != -1:
            start = max(idx - 50, 0)
            end = min(idx + len(query) + 50, len(text))
            snippet = text[start:end].replace('\n', ' ')
            return f"...{snippet}..."
        return text[:100] + '...'
    results['snippet'] = results['text'].apply(make_snippet)
    # Select relevant columns for the frontend
    columns = ['Docket No.', 'Granted', 'Argued', 'Decided', 'url', 'snippet']
    return results[columns].to_json(orient='records')

# oyez endpoint
@application.route('/oyez')
def oyez():
    year = request.args.get('year')
    docket = request.args.get('docket')
    oyez_url = ''
    if year and docket:
        oyez_url = f'https://api.oyez.org/cases/{year}/{docket}'
    return render_template('oyez.html', oyez_url=oyez_url)

# docs
@application.route('/docs')
def docs():
    return render_template('docs.html')

# about
@application.route('/about')
def about():
    return render_template('about.html')

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/api/oyez-summary', methods=['POST'])
def oyez_summary():
    data = request.json
    # Extract relevant fields from the posted data
    summary = data.get('summary', '')
    ideology_scores = data.get('ideology_scores', [])
    # Compose a prompt for DeepSeek
    prompt = f"""
    Given the following Supreme Court case summary and ideology scores of the justices, 
    write a concise, neutral summary that explains the ideological split and its significance.
    If ideological scores are all 0 then the data wasn't available.

    Summary:
    {summary}

    Ideology Scores:
    """
    for j in ideology_scores:
        prompt += f"- {j['name']}: {j['ideology']}\n"
    prompt += "\nRespond in 3-5 sentences."

    # Call AWS Bedrock DeepSeek (requires credentials and inference profile)
    try:
        profile_name = os.environ.get('AWS_PROFILE_NAME')
        if profile_name:
            session = boto3.Session(profile_name=profile_name)
        else:
            session = boto3.Session()
        bedrock = session.client("bedrock-runtime", region_name=os.environ.get('AWS_REGION', 'us-east-1'))
        print('DEEPSEEK_INFERENCE_ID:', os.environ.get('DEEPSEEK_INFERENCE_ID'))
        print('Prompt:', prompt)
        response = bedrock.invoke_model(
            modelId="us.deepseek.r1-v1:0",
            contentType="application/json",
            accept="application/json",
            body=f'{{"prompt": {json.dumps(prompt)}, "max_tokens": 256}}',
        )
        result = response['body'].read().decode()
        print('DeepSeek response:', result)
        return jsonify({'summary': result})
    except Exception as e:
        import traceback
        print('Exception in /api/oyez-summary:', traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@application.route('/kwic')
def kwic():
    query = request.args.get('query', '')
    window = int(request.args.get('window', 5))  # word window (legacy, for compatibility)
    left_chars = request.args.get('left_chars')
    right_chars = request.args.get('right_chars')
    # Default to None if not provided, else convert to int
    left_chars = int(left_chars) if left_chars is not None else None
    right_chars = int(right_chars) if right_chars is not None else None
    if not query:
        return jsonify({'error': 'No search query provided'}), 400
    results = []
    for text in df['text'].dropna():
        for match in re.finditer(re.escape(query), text, re.IGNORECASE):
            start_idx = match.start()
            end_idx = match.end()
            if left_chars is not None or right_chars is not None:
                # Character-based context window
                l_chars = left_chars if left_chars is not None else 50
                r_chars = right_chars if right_chars is not None else 50
                left = text[max(0, start_idx - l_chars):start_idx]
                keyword = text[start_idx:end_idx]
                right = text[end_idx:end_idx + r_chars]
                results.append({'left': left, 'keyword': keyword, 'right': right})
            else:
                # Word-based context window (legacy)
                words = re.findall(r'\w+|\W+', text)
                char_count = 0
                match_word_idx = None
                for i, w in enumerate(words):
                    char_count += len(w)
                    if char_count >= start_idx + 1:
                        match_word_idx = i
                        break
                if match_word_idx is not None:
                    left = ''.join(words[max(0, match_word_idx-window):match_word_idx])
                    kwic_word = ''.join(words[match_word_idx:match_word_idx+1])
                    right = ''.join(words[match_word_idx+1:match_word_idx+1+window])
                    results.append({'left': left.strip(), 'keyword': kwic_word.strip(), 'right': right.strip()})
    return jsonify(results)

@application.route('/collocates')
def collocates():
    query = request.args.get('query', '')
    window = int(request.args.get('window', 5))
    if not query:
        return jsonify({'error': 'No search query provided'}), 400
    colloc_counter = Counter()
    for text in df['text'].dropna():
        for match in re.finditer(re.escape(query), text, re.IGNORECASE):
            start_idx = match.start()
            end_idx = match.end()
            words = re.findall(r'\w+|\W+', text)
            char_count = 0
            match_word_idx = None
            for i, w in enumerate(words):
                char_count += len(w)
                if char_count >= start_idx + 1:
                    match_word_idx = i
                    break
            if match_word_idx is not None:
                left = words[max(0, match_word_idx-window):match_word_idx]
                right = words[match_word_idx+1:match_word_idx+1+window]
                for w in left + right:
                    if w.strip().isalpha() and w.lower() != query.lower():
                        colloc_counter[w.lower()] += 1
    top_collocates = colloc_counter.most_common(30)
    return jsonify([{'word': w, 'count': c} for w, c in top_collocates])

if __name__ == '__main__':
    application.run(debug=True)


