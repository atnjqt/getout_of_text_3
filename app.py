# A simple flask app that 
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

app = Flask(__name__)

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

@app.route('/data.json')
def data_json():
    # Return the full DataFrame as JSON
    return df.to_json(orient='records')

# give me a route that shows the df to_html with bootstrap styling
@app.route('/data.html')
def data_html():
    # Convert DataFrame to HTML with Bootstrap styling
    # show only one line of text per row
    df
    # Create a copy of the DataFrame with truncated text
    display_df = df.copy()
    display_df['text'] = display_df['text'].str[1000:1100] + '...'
    html_table = display_df.to_html(classes='table table-striped table-bordered', index=False)
    return render_template('data.html', table=html_table)

@app.route('/search', methods=['GET'])
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
@app.route('/oyez')
def oyez():
    year = request.args.get('year')
    docket = request.args.get('docket')
    oyez_url = ''
    if year and docket:
        oyez_url = f'https://api.oyez.org/cases/{year}/{docket}'
    return render_template('oyez.html', oyez_url=oyez_url)

# docs
@app.route('/docs')
def docs():
    return render_template('docs.html')

# about
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/')
def index():
    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)
