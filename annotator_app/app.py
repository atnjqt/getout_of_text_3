from flask import Flask, render_template, request, jsonify
import json
import os
import pandas as pd
from pathlib import Path
from datetime import datetime
import spacy
from spacy import displacy
from collections import Counter
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import re

# Langchain imports for AI agent
try:
    from langchain.agents import create_agent
    from langchain.tools import tool
    from langchain.chat_models import init_chat_model
    #from dotenv import load_dotenv
    #load_dotenv()
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️ Langchain not available. AI agent features will be disabled.")

app = Flask(__name__)

# Helper function to strip HTML tags
def strip_html_tags(text):
    """Remove HTML tags from text"""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

# Configure paths
DATA_DIR = Path(__file__).parent.parent / 'data'
EXPORTS_DIR = Path(__file__).parent.parent / 'exports'
ANNOTATIONS_DIR = Path(__file__).parent / 'annotations'
ANNOTATIONS_DIR.mkdir(exist_ok=True)
SETTINGS_FILE = Path(__file__).parent / 'settings.json'

# Configure upload settings
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'csv'}

# Configure upload settings
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def parse_sketch_engine_filename(filename):
    """
    Parse Sketch Engine filename to extract metadata.
    Format: concordance_{keyword}_{corpus}_{timestamp}.csv
    
    Examples:
        concordance_best_system_ecolexicon_en_20251115000951.csv
        concordance_best_system_oec_biwec3_20251117072438.csv
    
    Returns dict with 'keyword', 'corpus', 'timestamp' or None if unparseable
    """
    name = filename.replace('.csv', '')
    
    # Pattern: concordance_{keyword}_{corpus}_{timestamp}
    match = re.match(r'concordance_(.+)_(\d{14})$', name)
    
    if not match:
        return None
    
    keywords_and_corpus = match.group(1)
    timestamp = match.group(2)
    
    # Split and identify corpus
    parts = keywords_and_corpus.split('_')
    corpus_patterns = ['ecolexicon', 'oec', 'biwec', 'coca', 'coha', 'glowbe']
    
    corpus_idx = None
    for i, part in enumerate(parts):
        if any(pattern in part.lower() for pattern in corpus_patterns):
            corpus_idx = i
            break
    
    if corpus_idx is not None:
        keyword = '_'.join(parts[:corpus_idx])
        corpus = '_'.join(parts[corpus_idx:])
    else:
        keyword = '_'.join(parts[:-1])
        corpus = parts[-1]
    
    return {
        'keyword': keyword,
        'corpus': corpus,
        'timestamp': timestamp
    }

def convert_sketch_csv_to_kwic(csv_file, metadata):
    """
    Convert Sketch Engine CSV to KWIC JSON format.
    
    Args:
        csv_file: File object or path to CSV file
        metadata: Dict with 'keyword', 'corpus', 'timestamp'
    
    Returns:
        Dict in KWIC format: {genre_key: [items]}
    """
    # Read CSV, skipping first 4 rows of metadata
    df = pd.read_csv(csv_file, skiprows=4)
    
    # Extract genre from corpus name
    genre = metadata['corpus'].split('_')[0]
    
    # Build KWIC items
    kwic_items = []
    
    for idx, row in df.iterrows():
        # Get concordance components
        left = str(row.get('Left', '')).strip()
        kwic = str(row.get('KWIC', '')).strip()
        right = str(row.get('Right', '')).strip()
        reference = str(row.get('Reference', '')).strip()
        
        # Skip empty rows
        if not kwic or kwic == 'nan':
            continue
        
        # Clean Sketch Engine markup tags (sentence boundaries, etc.)
        # Remove <s>, </s>, <g/>, and other common tags
        left = re.sub(r'</?s>|<g/>', ' ', left).strip()
        kwic = re.sub(r'</?s>|<g/>', ' ', kwic).strip()
        right = re.sub(r'</?s>|<g/>', ' ', right).strip()
        
        # Normalize multiple spaces
        left = re.sub(r'\s+', ' ', left)
        kwic = re.sub(r'\s+', ' ', kwic)
        right = re.sub(r'\s+', ' ', right)
        
        # Create context with bold markers around keyword
        context = f"{left} **{kwic}** {right}"
        
        # Create full text (without markers)
        full_text = f"{left} {kwic} {right}"
        
        # Create KWIC item
        item = {
            'text_id': reference if reference and reference != 'nan' else f"{genre}_{idx}",
            'match': kwic,
            'context': context,
            'full_text': full_text
        }
        
        kwic_items.append(item)
    
    # Create genre key (format: genre_keyword)
    genre_key = f"{genre}_{metadata['keyword']}"
    
    return {genre_key: kwic_items}

def normalize_punctuation(text):
    """Normalize spacing around punctuation marks."""
    if not text:
        return text
    
    # Remove spaces before common punctuation marks
    text = re.sub(r'\s+([.,;:!?)])', r'\1', text)
    # Remove spaces after opening punctuation
    text = re.sub(r'([(])\s+', r'\1', text)
    # Fix multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Fix space before apostrophes in contractions
    text = re.sub(r"\s+'([st]|re|ve|ll|d|m)\b", r"'\1", text)
    # Fix n't contractions (don't, can't, won't, etc.)
    text = re.sub(r"\s+n't\b", r"n't", text)
    
    return text.strip()

# Load spaCy model (lazy loading)
_nlp = None
def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load('en_core_web_lg')
        except OSError:
            # Fallback to smaller model if large not available
            _nlp = spacy.load('en_core_web_sm')
    return _nlp

# Initialize Langchain agent (lazy loading)
_agent = None
_langchain_tools = []

if LANGCHAIN_AVAILABLE:
    @tool
    def get_morphology(text: str) -> str:
        """Get morphological analysis from the input text.
        
        Args:
            text: input text string
        Returns:
            morphology: list of tuples (token text, morphological features)
        """
        nlp = get_nlp()
        doc = nlp(text)
        morphology = [(token.text, str(token.morph)) for token in doc]
        return str(morphology)
    
    @tool
    def get_part_of_speech(text: str) -> str:
        """Get part of speech tags for the input text."""
        nlp = get_nlp()
        doc = nlp(text)
        pos_tags = [(token.text, token.pos_) for token in doc]
        return str(pos_tags)
    
    @tool
    def get_named_entities(text: str) -> str:
        """Get named entities from the input text."""
        nlp = get_nlp()
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return str(entities)
    
    @tool
    def get_most_frequent_ngrams(text: str, n: int=2, top: int=10) -> str:
        """Get the most frequent n-grams from the input text."""
        nlp = get_nlp()
        doc = nlp(text)
        tokens = [token.text for token in doc]
        # strip punctuation and lowercase
        tokens = [token.lower().strip('.,!?;"\'()[]{}') for token in tokens]
        tokens = [token for token in tokens if token]  # remove empty tokens
        ngrams = []
        for i in range(len(tokens)-n+1):
            ngrams.append(" ".join(tokens[i:i+n]))
        freq_dist = Counter(ngrams)
        most_common = freq_dist.most_common(top)
        return str(most_common)
    
    _langchain_tools = [
        get_morphology,
        get_part_of_speech,
        get_named_entities,
        get_most_frequent_ngrams
    ]

def get_agent():
    """Get or initialize the langchain agent."""
    global _agent
    
    if not LANGCHAIN_AVAILABLE:
        return None
    
    if _agent is None:
        try:
            # Initialize model - try AWS Bedrock first, fallback to other providers
            model_provider = os.getenv('MODEL_PROVIDER', 'bedrock_converse')
            model_id = os.getenv('MODEL_ID', 'openai.gpt-oss-120b-1:0')
            aws_profile = os.getenv('AWS_PROFILE', 'atn-developer')
            temperature = float(os.getenv('TEMPERATURE', '0.2'))
            max_tokens = int(os.getenv('MAX_TOKENS', '128000'))
            
            model = init_chat_model(
                model_id,
                model_provider=model_provider,
                credentials_profile_name=aws_profile,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Create agent
            _agent = create_agent(
                model=model,
                tools=_langchain_tools,
                system_prompt="You are a helpful assistant specializing in corpus linguistics analysis. "
                             "Your job is to provide information based on user queries using tools provided to facilitate "
                             "AI-assisted Corpus Linguistics keyword in context (KWIC) concordance data analysis. "
                             "Be concise and always use and trust the tools provided to get accurate information. "
                             "Never suggest information that is not based on tool outputs. "
                             "Focus on analyzing the morphological and linguistic features of keywords in their context."
            )
            
            print("✅ Langchain agent initialized successfully")
        except Exception as e:
            print(f"⚠️ Error initializing langchain agent: {e}")
            return None
    
    return _agent

def get_available_files():
    """Get list of JSON files in exports directory, sorted alphanumerically."""
    if not EXPORTS_DIR.exists():
        return []
    files = [f.name for f in EXPORTS_DIR.glob('*.json') if f.name.startswith('kwic_')]
    return sorted(files)

def load_kwic_data(filename):
    """Load KWIC data from JSON file and add cleaned text fields."""
    filepath = EXPORTS_DIR / filename
    if not filepath.exists():
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Add text_clean field to each KWIC item
    for genre_key, items in data.items():
        for item in items:
            # Clean context
            if 'context' in item:
                item['text_clean'] = normalize_punctuation(item['context'])
            # Clean full_text
            if 'full_text' in item:
                item['full_text_clean'] = normalize_punctuation(item['full_text'])
    
    return data

def load_annotations(filename):
    """Load existing annotations for a file."""
    annotation_file = ANNOTATIONS_DIR / f"{filename.replace('.json', '')}_annotations.json"
    if annotation_file.exists():
        with open(annotation_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def load_settings():
    """Load settings from settings.json, create default if doesn't exist."""
    if not SETTINGS_FILE.exists():
        default_settings = {
            'corpora': [
                {
                    'full_name': 'Corpus of Contemporary American English',
                    'friendly_name': 'COCA',
                    'relative_filepath': 'data/english-corpora.org/coca'
                },
                {
                    'full_name': 'Corpus of Historical American English',
                    'friendly_name': 'COHA',
                    'relative_filepath': 'data/english-corpora.org/coha'
                },
                {
                    'full_name': 'Corpus of Global Web-Based English',
                    'friendly_name': 'GLOWBE',
                    'relative_filepath': 'data/english-corpora.org/glowbe'
                },
                {
                    'full_name': 'EcoLexicon English Corpus',
                    'friendly_name': 'EcoLexicon',
                    'relative_filepath': 'data/sketch-engine'
                }
            ]
        }
        save_settings(default_settings)
        return default_settings
    
    with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_settings(settings):
    """Save settings to settings.json."""
    with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)

def load_settings():
    """Load settings from settings.json, create default if doesn't exist."""
    if not SETTINGS_FILE.exists():
        default_settings = {
            'corpora': [
                {
                    'full_name': 'Corpus of Contemporary American English',
                    'friendly_name': 'COCA',
                    'relative_filepath': 'data/english-corpora.org/coca'
                },
                {
                    'full_name': 'Corpus of Historical American English',
                    'friendly_name': 'COHA',
                    'relative_filepath': 'data/english-corpora.org/coha'
                },
                {
                    'full_name': 'Corpus of Global Web-Based English',
                    'friendly_name': 'GLOWBE',
                    'relative_filepath': 'data/english-corpora.org/glowbe'
                },
                {
                    'full_name': 'EcoLexicon English Corpus',
                    'friendly_name': 'EcoLexicon',
                    'relative_filepath': 'data/sketch-engine'
                }
            ]
        }
        save_settings(default_settings)
        return default_settings
    
    with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_settings(settings):
    """Save settings to settings.json."""
    with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)

def save_annotation(filename, genre_key, index, annotation_data):
    """Save annotation for a specific KWIC hit with full context."""
    annotation_file = ANNOTATIONS_DIR / f"{filename.replace('.json', '')}_annotations.json"
    
    # Load existing annotations
    annotations = load_annotations(filename)
    
    # Load KWIC data to get full context
    kwic_data = load_kwic_data(filename)
    
    # Create nested structure if needed
    if genre_key not in annotations:
        annotations[genre_key] = {}
    
    # Get the KWIC item to include context data
    kwic_item = {}
    if kwic_data and genre_key in kwic_data and index < len(kwic_data[genre_key]):
        kwic_item = kwic_data[genre_key][index]
    
    # Save annotation with timestamp and full context
    annotations[genre_key][str(index)] = {
        'text_id': kwic_item.get('text_id', ''),
        'match': kwic_item.get('match', ''),
        'context': kwic_item.get('context', ''),
        'full_text': kwic_item.get('full_text', ''),
        **annotation_data,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save to file
    with open(annotation_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    
    return True

@app.route('/')
def index():
    """Main annotation interface."""
    available_files = get_available_files()
    return render_template('index.html', files=available_files)

@app.route('/api/upload_csv', methods=['POST'])
def api_upload_csv():
    """API endpoint to upload and convert Sketch Engine CSV file."""
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    # Check if file has a filename
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    # Check file extension
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Only CSV files are allowed'}), 400
    
    try:
        # Parse filename to extract metadata
        metadata = parse_sketch_engine_filename(file.filename)
        
        if not metadata:
            return jsonify({
                'success': False, 
                'error': 'Could not parse filename. Expected format: concordance_{keyword}_{corpus}_{timestamp}.csv'
            }), 400
        
        # Convert CSV to KWIC format
        kwic_data = convert_sketch_csv_to_kwic(file, metadata)
        
        # Count total items
        total_items = sum(len(items) for items in kwic_data.values())
        
        if total_items == 0:
            return jsonify({
                'success': False,
                'error': 'No valid concordance lines found in CSV file'
            }), 400
        
        # Generate output filename
        output_filename = f"kwic_{metadata['keyword']}_{metadata['corpus']}.json"
        output_path = EXPORTS_DIR / output_filename
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(kwic_data, f, indent=2, ensure_ascii=False)
        
        return jsonify({
            'success': True,
            'message': f'Successfully converted and saved {output_filename}',
            'filename': output_filename,
            'total_items': total_items,
            'keyword': metadata['keyword'],
            'corpus': metadata['corpus']
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error processing file: {str(e)}'
        }), 500

@app.route('/api/files')
def api_files():
    """API endpoint to get available files."""
    files = get_available_files()
    return jsonify({'files': files})

@app.route('/api/load/<filename>')
def api_load(filename):
    """API endpoint to load KWIC data."""
    if filename not in get_available_files():
        return jsonify({'error': 'File not found'}), 404
    
    kwic_data = load_kwic_data(filename)
    if kwic_data is None:
        return jsonify({'error': 'Failed to load file'}), 500
    
    # Load existing annotations
    annotations = load_annotations(filename)
    
    # Count total items and annotated items
    total_items = sum(len(items) for items in kwic_data.values())
    annotated_items = sum(len(genre_annotations) for genre_annotations in annotations.values())
    
    return jsonify({
        'data': kwic_data,
        'annotations': annotations,
        'stats': {
            'total': total_items,
            'annotated': annotated_items,
            'remaining': total_items - annotated_items
        }
    })

@app.route('/api/save', methods=['POST'])
def api_save():
    """API endpoint to save an annotation."""
    data = request.json
    
    filename = data.get('filename')
    genre_key = data.get('genre_key')
    index = data.get('index')
    annotation = data.get('annotation')
    
    if not all([filename, genre_key is not None, index is not None, annotation]):
        return jsonify({'error': 'Missing required fields'}), 400
    
    if filename not in get_available_files():
        return jsonify({'error': 'Invalid file'}), 404
    
    success = save_annotation(filename, genre_key, index, annotation)
    
    if success:
        return jsonify({'success': True, 'message': 'Annotation saved'})
    else:
        return jsonify({'error': 'Failed to save annotation'}), 500

@app.route('/api/export/<filename>')
def api_export(filename):
    """API endpoint to export annotations with full context."""
    if filename not in get_available_files():
        return jsonify({'error': 'File not found'}), 404
    
    # Load both annotations and KWIC data
    annotations = load_annotations(filename)
    kwic_data = load_kwic_data(filename)
    
    if not kwic_data:
        return jsonify({'error': 'Could not load KWIC data'}), 500
    
    # Enrich annotations with context text
    enriched_annotations = {}
    for genre_key, genre_annotations in annotations.items():
        enriched_annotations[genre_key] = {}
        for idx_str, annotation in genre_annotations.items():
            idx = int(idx_str)
            # Get the corresponding KWIC item
            if genre_key in kwic_data and idx < len(kwic_data[genre_key]):
                item = kwic_data[genre_key][idx]
                enriched_annotations[genre_key][idx_str] = {
                    **annotation,
                    'text_id': item.get('text_id'),
                    'match': item.get('match'),
                    'context': item.get('context'),
                    'full_text': item.get('full_text')
                }
            else:
                # Include annotation even if KWIC item not found
                enriched_annotations[genre_key][idx_str] = annotation
    
    return jsonify({
        'filename': filename,
        'annotations': enriched_annotations,
        'exported_at': datetime.now().isoformat()
    })

@app.route('/api/spacy_analysis', methods=['POST'])
def api_spacy_analysis():
    """API endpoint to get spaCy morphological analysis."""
    data = request.json
    context_text = data.get('context', '')
    
    if not context_text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Remove markdown bold markers and normalize punctuation
    clean_text = normalize_punctuation(context_text.replace('**', ''))
    
    try:
        nlp = get_nlp()
        doc = nlp(clean_text)
        
        # Generate dependency visualization HTML
        dep_svg = displacy.render(doc, style='dep', options={
            'compact': False,
            'distance': 120,
            'arrow_stroke': 2,
            'arrow_width': 8
        })
        
        # Extract token information
        tokens = []
        for token in doc:
            tokens.append({
                'text': token.text,
                'lemma': token.lemma_,
                'pos': token.pos_,
                'tag': token.tag_,
                'dep': token.dep_,
                'head': token.head.text,
                'head_pos': token.head.pos_,
                'children': [child.text for child in token.children]
            })
        
        return jsonify({
            'success': True,
            'dep_svg': dep_svg,
            'tokens': tokens,
            'model': nlp.meta['name']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/agent_analysis', methods=['POST'])
def api_agent_analysis():
    """API endpoint to get AI agent morphological analysis of keyword."""
    if not LANGCHAIN_AVAILABLE:
        return jsonify({
            'success': False, 
            'error': 'Langchain not available. Please install: pip install langchain langchain-aws'
        }), 503
    
    data = request.json
    context_text = data.get('context', '')
    keyword = data.get('keyword', '')
    
    if not context_text:
        return jsonify({'success': False, 'error': 'No context text provided'}), 400
    
    # Remove markdown bold markers and normalize punctuation
    clean_text = normalize_punctuation(context_text.replace('**', ''))
    
    try:
        agent = get_agent()
        
        if agent is None:
            return jsonify({
                'success': False,
                'error': 'Agent initialization failed. Check AWS credentials and model configuration.'
            }), 500
        
        # Construct query for agent
        if keyword:
            query_text = f"Please provide a morphological analysis of what the speaker means by the phrase '{keyword}' in the full context of the following concordance data='{clean_text}'"
        else:
            query_text = f"Please provide a morphological analysis of what the speaker means in the full context of the following concordance data: {clean_text}"
        
        # Invoke agent
        result = agent.invoke({
            "messages": [{"role": "user", "content": query_text}]
        })
        
        # Extract the final text response from agent
        final_message = result['messages'][-1]
        
        # Handle different response formats
        if hasattr(final_message, 'content'):
            if isinstance(final_message.content, list):
                # Find text content in list
                text_response = None
                for item in final_message.content:
                    if isinstance(item, dict) and 'text' in item:
                        text_response = item['text']
                        break
                
                if text_response is None:
                    text_response = str(final_message.content)
            else:
                text_response = final_message.content
        else:
            text_response = str(final_message)
        
        return jsonify({
            'success': True,
            'analysis': text_response,
            'keyword': keyword
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f"Agent analysis failed: {str(e)}"
        }), 500

@app.route('/analysis')
def analysis():
    """Analysis dashboard for annotations."""
    annotation_files = [f.name for f in ANNOTATIONS_DIR.glob('*_annotations.json')]
    return render_template('analysis.html', files=annotation_files)

@app.route('/morphology')
def morphology():
    """Morphological analysis of annotations."""
    annotation_files = [f.name for f in ANNOTATIONS_DIR.glob('*_annotations.json')]
    return render_template('morphology.html', files=annotation_files)

@app.route('/part-of-speech')
def part_of_speech():
    """Detailed POS analysis of annotations."""
    annotation_files = [f.name for f in ANNOTATIONS_DIR.glob('*_annotations.json')]
    return render_template('part_of_speech.html', files=annotation_files)

@app.route('/n-grams')
def ngrams():
    """N-gram analysis of annotations."""
    annotation_files = [f.name for f in ANNOTATIONS_DIR.glob('*_annotations.json')]
    return render_template('ngrams.html', files=annotation_files)

@app.route('/subject_analysis')
def subject_analysis():
    """Syntactic role analysis of annotations."""
    annotation_files = [f.name for f in ANNOTATIONS_DIR.glob('*_annotations.json')]
    return render_template('subject_analysis.html', files=annotation_files)

@app.route('/clustering')
def clustering():
    """POS-based clustering analysis of annotations."""
    annotation_files = [f.name for f in ANNOTATIONS_DIR.glob('*_annotations.json')]
    return render_template('clustering.html', files=annotation_files)

@app.route('/frequencies')
def frequencies():
    """Genre-based frequency analysis of annotations."""
    annotation_files = [f.name for f in ANNOTATIONS_DIR.glob('*_annotations.json')]
    return render_template('frequencies.html', files=annotation_files)

@app.route('/ai-summary')
def ai_summary():
    """AI-powered KWIC analysis interface."""
    kwic_files = get_available_files()
    annotation_files = [f.name for f in ANNOTATIONS_DIR.glob('*_annotations.json')]
    return render_template('ai_summary.html', kwic_files=kwic_files, annotation_files=annotation_files)

@app.route('/migrate-annotations')
def migrate_annotations():
    """Utility page to migrate existing annotation files to include context data."""
    annotation_files = [f.name for f in ANNOTATIONS_DIR.glob('*_annotations.json')]
    return render_template('migrate_annotations.html', files=annotation_files)

@app.route('/settings')
def settings():
    """Settings page for managing corpus entries."""
    return render_template('settings.html')

@app.route('/semantic-embedding')
def semantic_embedding():
    """Semantic embedding analysis interface."""
    annotation_files = [f.name for f in ANNOTATIONS_DIR.glob('*_annotations.json')]
    return render_template('embedding.html', files=annotation_files)

@app.route('/api/settings', methods=['GET'])
def api_get_settings():
    """Get current settings."""
    try:
        settings = load_settings()
        return jsonify({'success': True, 'settings': settings})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/settings', methods=['POST'])
def api_save_settings():
    """Save settings."""
    try:
        settings = request.json
        save_settings(settings)
        return jsonify({'success': True, 'message': 'Settings saved successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/corpus', methods=['POST'])
def api_add_corpus():
    """Add a new corpus entry."""
    try:
        corpus_data = request.json
        settings = load_settings()
        
        if 'corpora' not in settings:
            settings['corpora'] = []
        
        settings['corpora'].append(corpus_data)
        save_settings(settings)
        
        return jsonify({'success': True, 'message': 'Corpus added successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/corpus/<int:index>', methods=['PUT'])
def api_update_corpus(index):
    """Update a corpus entry."""
    try:
        corpus_data = request.json
        settings = load_settings()
        
        if 'corpora' not in settings or index >= len(settings['corpora']):
            return jsonify({'error': 'Corpus not found'}), 404
        
        settings['corpora'][index] = corpus_data
        save_settings(settings)
        
        return jsonify({'success': True, 'message': 'Corpus updated successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/corpus/<int:index>', methods=['DELETE'])
def api_delete_corpus(index):
    """Delete a corpus entry."""
    try:
        settings = load_settings()
        
        if 'corpora' not in settings or index >= len(settings['corpora']):
            return jsonify({'error': 'Corpus not found'}), 404
        
        settings['corpora'].pop(index)
        save_settings(settings)
        
        return jsonify({'success': True, 'message': 'Corpus deleted successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/migrate_annotation/<filename>', methods=['POST'])
def api_migrate_annotation(filename):
    """API endpoint to migrate a single annotation file to include KWIC context."""
    annotation_path = ANNOTATIONS_DIR / filename
    if not annotation_path.exists():
        return jsonify({'error': 'Annotation file not found'}), 404
    
    # Get the corresponding KWIC file
    kwic_filename = filename.replace('_annotations.json', '.json')
    if kwic_filename not in get_available_files():
        return jsonify({'error': 'Source KWIC file not found'}), 404
    
    kwic_data = load_kwic_data(kwic_filename)
    if not kwic_data:
        return jsonify({'error': 'Failed to load KWIC data'}), 500
    
    # Load existing annotations
    with open(annotation_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # Track statistics
    updated_count = 0
    skipped_count = 0
    
    # Update each annotation with context data
    for genre_key, genre_annotations in annotations.items():
        for idx_str, annotation in genre_annotations.items():
            idx = int(idx_str)
            
            # Check if context already exists
            if 'context' in annotation and 'full_text' in annotation:
                skipped_count += 1
                continue
            
            # Get KWIC item
            if genre_key in kwic_data and idx < len(kwic_data[genre_key]):
                kwic_item = kwic_data[genre_key][idx]
                
                # Add missing fields
                annotation['text_id'] = annotation.get('text_id', kwic_item.get('text_id', ''))
                annotation['match'] = annotation.get('match', kwic_item.get('match', ''))
                annotation['context'] = annotation.get('context', kwic_item.get('context', ''))
                annotation['full_text'] = annotation.get('full_text', kwic_item.get('full_text', ''))
                
                updated_count += 1
    
    # Save updated annotations
    with open(annotation_path, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    
    return jsonify({
        'success': True,
        'filename': filename,
        'updated': updated_count,
        'skipped': skipped_count,
        'total': updated_count + skipped_count
    })

@app.route('/api/frequencies/<filename>')
def api_frequencies(filename):
    """API endpoint to get genre distribution by classification."""
    annotation_path = ANNOTATIONS_DIR / filename
    if not annotation_path.exists():
        return jsonify({'error': 'Annotation file not found'}), 404
    
    # Determine the source KWIC file
    kwic_filename = filename.replace('_annotations.json', '.json')
    if kwic_filename not in get_available_files():
        return jsonify({'error': 'Source KWIC file not found'}), 404
    
    kwic_data = load_kwic_data(kwic_filename)
    if kwic_data is None:
        return jsonify({'error': 'Failed to load KWIC file'}), 500
    
    annotations = load_annotations(kwic_filename)
    
    # Initialize counters
    genre_classification_counts = {}
    classification_totals = {'literal': 0, 'figurative': 0, 'neither': 0, 'unclear': 0}
    genre_totals = {}
    
    # Process each annotated item
    for genre_key, items in kwic_data.items():
        # Extract just the genre name (before the underscore)
        genre_name = genre_key.split('_')[0]
        
        if genre_name not in genre_classification_counts:
            genre_classification_counts[genre_name] = {
                'literal': 0,
                'figurative': 0,
                'neither': 0,
                'unclear': 0,
                'total': 0
            }
        
        for idx, item in enumerate(items):
            annotation = annotations.get(genre_key, {}).get(str(idx), {})
            if not annotation:
                continue
            
            classification = annotation.get('classification')
            if not classification or classification not in classification_totals:
                continue
            
            genre_classification_counts[genre_name][classification] += 1
            genre_classification_counts[genre_name]['total'] += 1
            classification_totals[classification] += 1
    
    # Calculate genre totals
    for genre in genre_classification_counts:
        genre_totals[genre] = genre_classification_counts[genre]['total']
    
    # Calculate percentages
    genre_percentages = {}
    for genre, counts in genre_classification_counts.items():
        genre_percentages[genre] = {}
        total = counts['total']
        if total > 0:
            for classification in ['literal', 'figurative', 'neither', 'unclear']:
                genre_percentages[genre][classification] = round((counts[classification] / total) * 100, 1)
    
    # Sort genres by total count
    sorted_genres = sorted(genre_classification_counts.keys(), 
                          key=lambda x: genre_classification_counts[x]['total'], 
                          reverse=True)
    
    return jsonify({
        'filename': filename,
        'genre_counts': genre_classification_counts,
        'genre_percentages': genre_percentages,
        'classification_totals': classification_totals,
        'genre_totals': genre_totals,
        'sorted_genres': sorted_genres
    })

@app.route('/api/clustering/<filename>')
def api_clustering(filename):
    """API endpoint to perform POS-based clustering analysis by classification."""
    annotation_path = ANNOTATIONS_DIR / filename
    if not annotation_path.exists():
        return jsonify({'error': 'Annotation file not found'}), 404
    
    # Determine the source KWIC file
    kwic_filename = filename.replace('_annotations.json', '.json')
    if kwic_filename not in get_available_files():
        return jsonify({'error': 'Source KWIC file not found'}), 404
    
    kwic_data = load_kwic_data(kwic_filename)
    if kwic_data is None:
        return jsonify({'error': 'Failed to load KWIC file'}), 500
    
    annotations = load_annotations(kwic_filename)
    
    # Get mode parameter (annotated vs raw)
    mode = request.args.get('mode', 'annotated')
    
    try:
        nlp = get_nlp()
        
        # Organize data by classification
        classification_data = {
            'literal': [],
            'figurative': [],
            'neither': [],
            'unclear': []
        }
        
        if mode == 'raw':
            # Process all KWIC items regardless of annotation
            classification_data['all'] = []
            
            for genre_key, items in kwic_data.items():
                for idx, item in enumerate(items):
                    # Use cleaned text if available
                    context = item.get('text_clean', item.get('context', ''))
                    # Strip HTML tags and remove ** markers
                    clean_text = strip_html_tags(context).replace('**', '')
                    
                    # Check if this item has an annotation
                    annotation = annotations.get(genre_key, {}).get(str(idx), {})
                    annotation_class = annotation.get('classification', 'unannotated')
                    annotation_notes = annotation.get('notes', '')
                    
                    # Analyze with spaCy
                    doc = nlp(clean_text)
                    
                    # Extract POS tags
                    pos_tags = [token.pos_ for token in doc]
                    
                    # Extract POS bigrams
                    pos_bigrams = [f"{pos_tags[i]}_{pos_tags[i+1]}" for i in range(len(pos_tags)-1)]
                    
                    # Extract POS trigrams
                    pos_trigrams = [f"{pos_tags[i]}_{pos_tags[i+1]}_{pos_tags[i+2]}" for i in range(len(pos_tags)-2)]
                    
                    classification_data['all'].append({
                        'text': context,
                        'genre': genre_key,
                        'text_id': item.get('text_id', ''),
                        'pos_tags': pos_tags,
                        'pos_bigrams': pos_bigrams,
                        'pos_trigrams': pos_trigrams,
                        'annotation_class': annotation_class,
                        'notes': annotation_notes
                    })
        else:
            # Process only annotated items
            for genre_key, items in kwic_data.items():
                for idx, item in enumerate(items):
                    annotation = annotations.get(genre_key, {}).get(str(idx), {})
                    if not annotation:
                        continue
                    
                    classification = annotation.get('classification')
                    if not classification or classification not in classification_data:
                        continue
                    
                    # Use cleaned text if available
                    context = item.get('text_clean', item.get('context', ''))
                    # Strip HTML tags and remove ** markers
                    clean_text = strip_html_tags(context).replace('**', '')
                    
                    # Analyze with spaCy
                    doc = nlp(clean_text)
                    
                    # Extract POS tags
                    pos_tags = [token.pos_ for token in doc]
                    
                    # Extract POS bigrams
                    pos_bigrams = [f"{pos_tags[i]}_{pos_tags[i+1]}" for i in range(len(pos_tags)-1)]
                    
                    # Extract POS trigrams
                    pos_trigrams = [f"{pos_tags[i]}_{pos_tags[i+1]}_{pos_tags[i+2]}" for i in range(len(pos_tags)-2)]
                    
                    classification_data[classification].append({
                        'text': context,
                        'genre': genre_key,
                        'text_id': item.get('text_id', ''),
                        'pos_tags': pos_tags,
                        'pos_bigrams': pos_bigrams,
                        'pos_trigrams': pos_trigrams,
                        'notes': annotation.get('notes', '')
                    })
        
        # Perform clustering for each classification
        results = {}
        for classification, items in classification_data.items():
            if len(items) < 2:
                results[classification] = {
                    'error': f'Not enough data for clustering (need at least 2 items, have {len(items)})',
                    'count': len(items)
                }
                continue
            
            # Feature engineering: POS tag frequency vectors
            all_pos_tags = set()
            all_bigrams = set()
            all_trigrams = set()
            
            for item in items:
                all_pos_tags.update(item['pos_tags'])
                all_bigrams.update(item['pos_bigrams'])
                all_trigrams.update(item['pos_trigrams'])
            
            # Create feature vectors
            feature_vectors = []
            for item in items:
                # Count POS tags
                pos_counts = Counter(item['pos_tags'])
                tag_vector = [pos_counts.get(tag, 0) for tag in sorted(all_pos_tags)]
                
                # Count bigrams
                bigram_counts = Counter(item['pos_bigrams'])
                bigram_vector = [bigram_counts.get(bg, 0) for bg in sorted(all_bigrams)]
                
                # Count trigrams
                trigram_counts = Counter(item['pos_trigrams'])
                trigram_vector = [trigram_counts.get(tg, 0) for tg in sorted(all_trigrams)]
                
                # Combine features
                feature_vectors.append(tag_vector + bigram_vector + trigram_vector)
            
            # Convert to numpy array
            X = np.array(feature_vectors)
            
            # Normalize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Determine optimal number of clusters (use elbow method heuristic)
            n_samples = len(items)
            n_clusters = min(5, max(2, n_samples // 3))  # Between 2 and 5 clusters
            
            # Perform K-Means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Organize results by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                label = int(label)
                if label not in clusters:
                    clusters[label] = []
                
                item_data = {
                    'text': items[i]['text'],
                    'genre': items[i]['genre'],
                    'text_id': items[i]['text_id'],
                    'pos_sequence': ' '.join(items[i]['pos_tags']),
                    'notes': items[i].get('notes', '')
                }
                
                # Include annotation info if in raw mode
                if 'annotation_class' in items[i]:
                    item_data['annotation_class'] = items[i]['annotation_class']
                
                clusters[label].append(item_data)
            
            # Calculate cluster statistics
            cluster_stats = []
            for label in sorted(clusters.keys()):
                cluster_items = clusters[label]
                # Most common POS patterns in this cluster
                all_cluster_tags = []
                all_cluster_bigrams = []
                all_cluster_trigrams = []
                for idx in [i for i, l in enumerate(cluster_labels) if l == label]:
                    all_cluster_tags.extend(items[idx]['pos_tags'])
                    all_cluster_bigrams.extend(items[idx]['pos_bigrams'])
                    all_cluster_trigrams.extend(items[idx]['pos_trigrams'])
                
                top_tags = Counter(all_cluster_tags).most_common(5)
                top_bigrams = Counter(all_cluster_bigrams).most_common(5)
                top_trigrams = Counter(all_cluster_trigrams).most_common(5)
                
                cluster_stats.append({
                    'label': label,
                    'size': len(cluster_items),
                    'top_tags': [{'tag': tag, 'count': count} for tag, count in top_tags],
                    'top_bigrams': [{'bigram': bg, 'count': count} for bg, count in top_bigrams],
                    'top_trigrams': [{'trigram': tg, 'count': count} for tg, count in top_trigrams],
                    'items': cluster_items
                })
            
            results[classification] = {
                'count': len(items),
                'n_clusters': n_clusters,
                'clusters': cluster_stats
            }
        
        return jsonify({
            'filename': filename,
            'mode': mode,
            'results': results,
            'model': nlp.meta['name']
        })
        
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/subject_analysis/<filename>')
def api_subject_analysis(filename):
    """API endpoint to get syntactic role counts for 'best system' by classification."""
    annotation_path = ANNOTATIONS_DIR / filename
    if not annotation_path.exists():
        return jsonify({'error': 'Annotation file not found'}), 404
    
    # Determine the source KWIC file
    kwic_filename = filename.replace('_annotations.json', '.json')
    if kwic_filename not in get_available_files():
        return jsonify({'error': 'Source KWIC file not found'}), 404
    
    kwic_data = load_kwic_data(kwic_filename)
    if kwic_data is None:
        return jsonify({'error': 'Failed to load KWIC file'}), 500
    
    annotations = load_annotations(kwic_filename)
    
    try:
        nlp = get_nlp()
        
        # Initialize counters for each classification
        results = {
            'literal': {'subject': 0, 'indirect_object': 0, 'direct_object': 0, 'complement': 0, 'adjunct': 0, 'total': 0},
            'figurative': {'subject': 0, 'indirect_object': 0, 'direct_object': 0, 'complement': 0, 'adjunct': 0, 'total': 0},
            'neither': {'subject': 0, 'indirect_object': 0, 'direct_object': 0, 'complement': 0, 'adjunct': 0, 'total': 0},
            'unclear': {'subject': 0, 'indirect_object': 0, 'direct_object': 0, 'complement': 0, 'adjunct': 0, 'total': 0}
        }
        
        # Define dependency mappings for syntactic roles
        subject_deps = {'nsubj', 'nsubjpass', 'csubj', 'csubjpass'}
        direct_object_deps = {'dobj', 'ccomp', 'xcomp'}
        indirect_object_deps = {'iobj', 'dative'}
        complement_deps = {'acomp', 'attr'}
        adjunct_deps = {'advmod', 'npadvmod', 'prep', 'advcl', 'obl', 'nmod'}
        
        # Process each annotated item
        for genre_key, items in kwic_data.items():
            for idx, item in enumerate(items):
                annotation = annotations.get(genre_key, {}).get(str(idx), {})
                if not annotation:
                    continue
                
                classification = annotation.get('classification')
                if not classification or classification not in results:
                    continue
                
                results[classification]['total'] += 1
                
                # Use cleaned text if available
                context = item.get('text_clean', item.get('context', ''))
                # Find the bolded keyword in original context
                keyword_match = context.replace('**', '')
                
                # Analyze with spaCy
                doc = nlp(keyword_match)
                
                # Find "best system" tokens - look for "system" token and check if preceded by "best"
                best_system_token = None
                for i, token in enumerate(doc):
                    if token.text.lower() == 'system':
                        # Check if previous token is "best"
                        if i > 0 and doc[i-1].text.lower() == 'best':
                            best_system_token = token
                            break
                
                # If we found "best system", analyze only the "system" token's dependency
                if best_system_token:
                    dep = best_system_token.dep_.lower()
                    
                    if dep in subject_deps:
                        results[classification]['subject'] += 1
                    elif dep in direct_object_deps:
                        results[classification]['direct_object'] += 1
                    elif dep in indirect_object_deps:
                        results[classification]['indirect_object'] += 1
                    elif dep in complement_deps:
                        results[classification]['complement'] += 1
                    elif dep in adjunct_deps:
                        results[classification]['adjunct'] += 1
        
        return jsonify({
            'filename': filename,
            'results': results,
            'model': nlp.meta['name']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/part_of_speech/<filename>')
def api_part_of_speech(filename):
    """API endpoint to get aggregate POS analysis by classification or across all values."""
    annotation_path = ANNOTATIONS_DIR / filename
    if not annotation_path.exists():
        return jsonify({'error': 'Annotation file not found'}), 404
    
    # Check if we should analyze across all values
    across_all = request.args.get('across_all', 'false').lower() == 'true'
    
    # Determine the source KWIC file
    kwic_filename = filename.replace('_annotations.json', '.json')
    if kwic_filename not in get_available_files():
        return jsonify({'error': 'Source KWIC file not found'}), 404
    
    kwic_data = load_kwic_data(kwic_filename)
    if kwic_data is None:
        return jsonify({'error': 'Failed to load KWIC file'}), 500
    
    annotations = load_annotations(kwic_filename)
    
    try:
        nlp = get_nlp()
        
        if across_all:
            # Analyze all values together
            all_tokens = {}
            total_count = 0
            
            for genre_key, items in kwic_data.items():
                for idx, item in enumerate(items):
                    annotation = annotations.get(genre_key, {}).get(str(idx), {})
                    if not annotation:
                        continue
                    
                    classification = annotation.get('classification')
                    if not classification:
                        continue
                    
                    total_count += 1
                    
                    # Use cleaned text if available
                    context = item.get('text_clean', item.get('context', ''))
                    # Remove markdown bold markers
                    clean_text = context.replace('**', '')
                    
                    # Analyze with spaCy
                    doc = nlp(clean_text)
                    
                    # Aggregate token information
                    for token in doc:
                        # Create a unique key for this token combination
                        key = (token.text.lower(), token.lemma_, token.pos_, token.tag_, token.dep_)
                        
                        if key not in all_tokens:
                            all_tokens[key] = {
                                'text': token.text,
                                'lemma': token.lemma_,
                                'pos': token.pos_,
                                'tag': token.tag_,
                                'dep': token.dep_,
                                'count': 0
                            }
                        
                        all_tokens[key]['count'] += 1
            
            # Convert to sorted list
            sorted_tokens = sorted(
                all_tokens.values(),
                key=lambda x: x['count'],
                reverse=True
            )
            
            return jsonify({
                'filename': filename,
                'across_all': True,
                'results': {'all': sorted_tokens},
                'counts': {'all': total_count},
                'model': nlp.meta['name']
            })
        else:
            # Original behavior: analyze by classification
            # Initialize aggregate counters for each classification
            aggregates = {
                'literal': {},
                'figurative': {},
                'neither': {},
                'unclear': {}
            }
            
            # Count totals
            counts = {
                'literal': 0,
                'figurative': 0,
                'neither': 0,
                'unclear': 0
            }
            
            # Process each annotated item
            for genre_key, items in kwic_data.items():
                for idx, item in enumerate(items):
                    annotation = annotations.get(genre_key, {}).get(str(idx), {})
                    if not annotation:
                        continue
                    
                    classification = annotation.get('classification')
                    if not classification or classification not in aggregates:
                        continue
                    
                    counts[classification] += 1
                    
                    # Use cleaned text if available
                    context = item.get('text_clean', item.get('context', ''))
                    # Remove markdown bold markers
                    clean_text = context.replace('**', '')
                    
                    # Analyze with spaCy
                    doc = nlp(clean_text)
                    
                    # Aggregate token information
                    for token in doc:
                        # Create a unique key for this token combination
                        key = (token.text.lower(), token.lemma_, token.pos_, token.tag_, token.dep_)
                        
                        if key not in aggregates[classification]:
                            aggregates[classification][key] = {
                                'text': token.text,
                                'lemma': token.lemma_,
                                'pos': token.pos_,
                                'tag': token.tag_,
                                'dep': token.dep_,
                                'count': 0
                            }
                        
                        aggregates[classification][key]['count'] += 1
            
            # Convert aggregates to sorted lists
            results = {}
            for classification, tokens in aggregates.items():
                results[classification] = sorted(
                    tokens.values(),
                    key=lambda x: x['count'],
                    reverse=True
                )
            
            return jsonify({
                'filename': filename,
                'across_all': False,
                'results': results,
                'counts': counts,
                'model': nlp.meta['name']
            })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/morphology/<filename>')
def api_morphology(filename):
    """API endpoint to analyze POS patterns by classification."""
    annotation_path = ANNOTATIONS_DIR / filename
    if not annotation_path.exists():
        return jsonify({'error': 'Annotation file not found. Please annotate some KWIC hits first.'}), 404
    
    # Determine the source KWIC file
    kwic_filename = filename.replace('_annotations.json', '.json')
    if kwic_filename not in get_available_files():
        return jsonify({'error': 'Source KWIC file not found'}), 404
    
    kwic_data = load_kwic_data(kwic_filename)
    if kwic_data is None:
        return jsonify({'error': 'Failed to load KWIC file'}), 500
    
    annotations = load_annotations(kwic_filename)
    
    # Check if there are any annotations at all
    total_annotations = sum(len(items) for items in annotations.values())
    if total_annotations == 0:
        return jsonify({'error': 'No annotations found in this file. Please annotate some KWIC hits first.'}), 400
    
    try:
        nlp = get_nlp()
        
        # Initialize results structure
        results = {
            'literal': {'pre': {}, 'post': {}, 'count': 0},
            'figurative': {'pre': {}, 'post': {}, 'count': 0},
            'neither': {'pre': {}, 'post': {}, 'count': 0},
            'unclear': {'pre': {}, 'post': {}, 'count': 0}
        }
        
        # Process each annotated item
        for genre_key, items in kwic_data.items():
            for idx, item in enumerate(items):
                annotation = annotations.get(genre_key, {}).get(str(idx), {})
                if not annotation:
                    continue
                
                classification = annotation.get('classification')
                if not classification or classification not in results:
                    continue
                
                results[classification]['count'] += 1
                
                try:
                    # Use cleaned text if available, otherwise fall back to context
                    context = item.get('text_clean', item.get('context', ''))
                    match = item.get('match', '')
                    
                    if not context or not match:
                        continue
                    
                    # Try to split by keyword markers first (for legacy data)
                    if '**' in context:
                        parts = context.split('**')
                        if len(parts) >= 3:
                            left_context = parts[0].strip()
                            keyword = parts[1].strip()
                            right_context = parts[2].strip()
                        else:
                            continue
                    else:
                        # Find the match in the context
                        match_lower = match.lower()
                        context_lower = context.lower()
                        match_idx = context_lower.find(match_lower)
                        
                        if match_idx == -1:
                            # Try case-sensitive match
                            match_idx = context.find(match)
                        
                        if match_idx == -1:
                            continue
                        
                        # Split context around the match
                        left_context = context[:match_idx].strip()
                        keyword = context[match_idx:match_idx + len(match)].strip()
                        right_context = context[match_idx + len(match):].strip()
                    
                    # Analyze left context (pre-modifiers)
                    if left_context:
                        doc_left = nlp(left_context)
                        # Get last 3 tokens before keyword
                        tokens = list(doc_left)[-3:]
                        for token in tokens:
                            pos = token.pos_
                            results[classification]['pre'][pos] = results[classification]['pre'].get(pos, 0) + 1
                    
                    # Analyze right context (post-modifiers)
                    if right_context:
                        doc_right = nlp(right_context)
                        # Get first 3 tokens after keyword
                        tokens = list(doc_right)[:3]
                        for token in tokens:
                            pos = token.pos_
                            results[classification]['post'][pos] = results[classification]['post'].get(pos, 0) + 1
                except Exception as e:
                    # Skip items that fail to process
                    continue
        
        # Check if we processed any valid data
        total_processed = sum(data['count'] for data in results.values())
        if total_processed == 0:
            return jsonify({'error': 'No valid annotated items could be processed. Make sure your KWIC data has context with keyword markers (**keyword**).'}), 400
        
        # Convert to sorted lists for easier frontend consumption
        formatted_results = {}
        for classification, data in results.items():
            formatted_results[classification] = {
                'count': data['count'],
                'pre': sorted(data['pre'].items(), key=lambda x: x[1], reverse=True),
                'post': sorted(data['post'].items(), key=lambda x: x[1], reverse=True)
            }
        
        return jsonify({
            'filename': filename,
            'results': formatted_results,
            'model': nlp.meta['name']
        })
        
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/display')
def display():
    """KWIC concordance display."""
    kwic_files = get_available_files()
    annotation_files = [f.name for f in ANNOTATIONS_DIR.glob('*_annotations.json')]
    return render_template('display.html', kwic_files=kwic_files, annotation_files=annotation_files)

@app.route('/api/display/<filename>')
def api_display(filename):
    """API endpoint to get KWIC data formatted for concordance display."""
    if filename not in get_available_files():
        return jsonify({'error': 'File not found'}), 404
    
    kwic_data = load_kwic_data(filename)
    if kwic_data is None:
        return jsonify({'error': 'Failed to load file'}), 500
    
    # Flatten and format KWIC items for display
    concordance_lines = []
    
    for genre_key, items in kwic_data.items():
        for idx, item in enumerate(items):
            # Try to get pre-split context fields first
            if 'left' in item and 'right' in item:
                left_context = item.get('left', '')
                keyword = item.get('match', item.get('keyword', ''))
                right_context = item.get('right', '')
            else:
                # Use cleaned text if available
                context = item.get('text_clean', item.get('context', ''))
                match = item.get('match', '')
                
                # Split context by the keyword (marked with **)
                parts = context.split('**')
                
                if len(parts) >= 3:
                    left_context = parts[0].strip()
                    keyword = parts[1].strip()
                    right_context = parts[2].strip()
                else:
                    # Fallback: try to find the match keyword in context
                    keyword = match
                    if match and match in context:
                        # Find first occurrence of match in context
                        match_pos = context.find(match)
                        left_context = context[:match_pos].strip()
                        right_context = context[match_pos + len(match):].strip()
                    else:
                        # Can't split, put everything in right context
                        left_context = ''
                        right_context = context
            
            concordance_lines.append({
                'genre': genre_key,
                'text_id': item.get('text_id', ''),
                'left': left_context,
                'keyword': keyword,
                'right': right_context,
                'full_text': item.get('full_text', ''),
                'classification': None,
                'notes': None
            })
    
    return jsonify({
        'total': len(concordance_lines),
        'lines': concordance_lines
    })

@app.route('/api/display_annotations/<filename>')
def api_display_annotations(filename):
    """API endpoint to get annotated KWIC data formatted for concordance display."""
    # Load the annotation file
    annotation_path = ANNOTATIONS_DIR / filename
    if not annotation_path.exists():
        return jsonify({'error': 'Annotation file not found'}), 404
    
    # Determine the source KWIC file
    kwic_filename = filename.replace('_annotations.json', '.json')
    if kwic_filename not in get_available_files():
        return jsonify({'error': 'Source KWIC file not found'}), 404
    
    kwic_data = load_kwic_data(kwic_filename)
    if kwic_data is None:
        return jsonify({'error': 'Failed to load KWIC file'}), 500
    
    annotations = load_annotations(kwic_filename)
    
    # Flatten and format annotated KWIC items
    concordance_lines = []
    
    for genre_key, items in kwic_data.items():
        for idx, item in enumerate(items):
            # Check if this item has an annotation
            annotation = annotations.get(genre_key, {}).get(str(idx), {})
            
            # Only include annotated items
            if not annotation:
                continue
            
            # Try to get pre-split context fields first
            if 'left' in item and 'right' in item:
                left_context = item.get('left', '')
                keyword = item.get('match', item.get('keyword', ''))
                right_context = item.get('right', '')
            else:
                # Use cleaned text if available
                context = item.get('text_clean', item.get('context', ''))
                match = item.get('match', '')
                
                # Split context by the keyword (marked with **)
                parts = context.split('**')
                
                if len(parts) >= 3:
                    left_context = parts[0].strip()
                    keyword = parts[1].strip()
                    right_context = parts[2].strip()
                else:
                    # Fallback: try to find the match keyword in context
                    keyword = match
                    if match and match in context:
                        # Find first occurrence of match in context
                        match_pos = context.find(match)
                        left_context = context[:match_pos].strip()
                        right_context = context[match_pos + len(match):].strip()
                    else:
                        # Can't split, put everything in right context
                        left_context = ''
                        right_context = context
            
            concordance_lines.append({
                'genre': genre_key,
                'text_id': item.get('text_id', ''),
                'left': left_context,
                'keyword': keyword,
                'right': right_context,
                'full_text': item.get('full_text', ''),
                'classification': annotation.get('classification'),
                'notes': annotation.get('notes'),
                'timestamp': annotation.get('timestamp')
            })
    
    return jsonify({
        'total': len(concordance_lines),
        'lines': concordance_lines
    })

@app.route('/api/analysis/<filename>')
def api_analysis(filename):
    """API endpoint to get analysis data for a specific annotation file."""
    filepath = ANNOTATIONS_DIR / filename
    
    if not filepath.exists():
        return jsonify({'error': 'Annotation file not found'}), 404
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        # Count classifications
        classification_counts = {
            'literal': 0,
            'figurative': 0,
            'neither': 0,
            'unclear': 0
        }
        
        total_annotations = 0
        genre_breakdown = {}
        
        for genre_key, genre_annotations in annotations.items():
            genre_breakdown[genre_key] = {
                'literal': 0,
                'figurative': 0,
                'neither': 0,
                'unclear': 0,
                'total': 0
            }
            
            for annotation_data in genre_annotations.values():
                classification = annotation_data.get('classification', 'unclear')
                classification_counts[classification] += 1
                genre_breakdown[genre_key][classification] += 1
                genre_breakdown[genre_key]['total'] += 1
                total_annotations += 1
        
        # Calculate percentages
        percentages = {}
        if total_annotations > 0:
            for classification, count in classification_counts.items():
                percentages[classification] = round((count / total_annotations) * 100, 1)
        
        return jsonify({
            'filename': filename,
            'total_annotations': total_annotations,
            'classification_counts': classification_counts,
            'percentages': percentages,
            'genre_breakdown': genre_breakdown
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ngrams', methods=['POST'])
def api_ngrams():
    """API endpoint to generate n-grams from annotated data."""
    data = request.json
    
    kwic_filename = data.get('kwic_filename')
    classifications = data.get('classifications', ['literal', 'figurative', 'neither', 'unclear'])
    n_value = int(data.get('n', 2))  # Default bigrams
    min_freq = int(data.get('min_freq', 2))  # Minimum frequency to show
    
    if not kwic_filename:
        return jsonify({'error': 'No file specified'}), 400
    
    # Remove _annotations.json suffix if present
    if kwic_filename.endswith('_annotations.json'):
        kwic_filename = kwic_filename.replace('_annotations.json', '.json')
    
    # Load KWIC data
    kwic_data = load_kwic_data(kwic_filename)
    if not kwic_data:
        return jsonify({'error': 'Failed to load KWIC data'}), 500
    
    # Load annotations
    annotations = load_annotations(kwic_filename)
    
    try:
        nlp = get_nlp()
        
        # Collect texts by classification
        word_ngrams = {cls: Counter() for cls in classifications}
        lemma_ngrams = {cls: Counter() for cls in classifications}
        
        for genre_key, items in kwic_data.items():
            for idx, item in enumerate(items):
                # Get annotation for this item
                annotation = annotations.get(genre_key, {}).get(str(idx), {})
                classification = annotation.get('classification')
                
                # Skip if not annotated or not in selected classifications
                if not classification or classification not in classifications:
                    continue
                
                # Use cleaned full text if available, otherwise fall back to full_text
                full_text = item.get('full_text_clean', item.get('full_text', ''))
                if not full_text:
                    continue
                
                doc = nlp(full_text)
                
                # Extract word n-grams (excluding punctuation)
                tokens = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]
                for i in range(len(tokens) - n_value + 1):
                    ngram = tuple(tokens[i:i + n_value])
                    word_ngrams[classification][ngram] += 1
                
                # Extract lemma n-grams (excluding punctuation)
                lemmas = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
                for i in range(len(lemmas) - n_value + 1):
                    ngram = tuple(lemmas[i:i + n_value])
                    lemma_ngrams[classification][ngram] += 1
        
        # Filter by minimum frequency and format results
        word_results = {}
        lemma_results = {}
        
        for cls in classifications:
            # Word n-grams
            word_filtered = {
                ' '.join(ngram): count 
                for ngram, count in word_ngrams[cls].items() 
                if count >= min_freq
            }
            word_results[cls] = sorted(
                word_filtered.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Lemma n-grams
            lemma_filtered = {
                ' '.join(ngram): count 
                for ngram, count in lemma_ngrams[cls].items() 
                if count >= min_freq
            }
            lemma_results[cls] = sorted(
                lemma_filtered.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
        
        return jsonify({
            'success': True,
            'n': n_value,
            'min_freq': min_freq,
            'word_ngrams': word_results,
            'lemma_ngrams': lemma_results,
            'total_annotations': sum(
                len(genre_annotations) 
                for genre_annotations in annotations.values()
            )
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai_summary', methods=['POST'])
def api_ai_summary():
    """API endpoint to run AI analysis on KWIC data."""
    import time
    start_time = time.time()
    print(f"\n{'='*80}")
    print(f"[AI SUMMARY] Starting AI summary request at {datetime.now().isoformat()}")
    
    data = request.json
    
    filename = data.get('filename')
    reasoning_level = data.get('reasoning_level', 'high')
    corpus_fullname = data.get('corpus_fullname', '')
    corpus_shortname = data.get('corpus_shortname', '')
    keyword = data.get('keyword', '')
    random_sample = int(data.get('random_sample', 50))
    temperature = float(data.get('temperature', 0.0))
    aws_profile = data.get('aws_profile', 'default')
    remove_annotations = data.get('remove_annotations', True)
    
    print(f"[AI SUMMARY] Parameters: filename={filename}, keyword={keyword}, sample={random_sample}")
    
    if not filename:
        return jsonify({'error': 'No file selected'}), 400
    
    # Determine if it's a KWIC file or annotation file
    load_start = time.time()
    print(f"[AI SUMMARY] Loading data from {filename}...")
    
    if filename.endswith('_annotations.json'):
        # Load from annotation file
        annotation_path = ANNOTATIONS_DIR / filename
        if not annotation_path.exists():
            return jsonify({'error': 'Annotation file not found'}), 404
        
        # Get the corresponding KWIC file
        kwic_filename = filename.replace('_annotations.json', '.json')
        if kwic_filename not in get_available_files():
            return jsonify({'error': 'Source KWIC file not found'}), 404
        
        kwic_data = load_kwic_data(kwic_filename)
        annotations = load_annotations(kwic_filename)
        
        # Merge annotations back into kwic_data
        for genre_key, items in kwic_data.items():
            for idx, item in enumerate(items):
                annotation = annotations.get(genre_key, {}).get(str(idx), {})
                if annotation:
                    item.update(annotation)
    else:
        # Load directly from KWIC file
        if filename not in get_available_files():
            return jsonify({'error': 'KWIC file not found'}), 404
        
        kwic_data = load_kwic_data(filename)
    
    if not kwic_data:
        return jsonify({'error': 'Failed to load data'}), 500
    
    load_time = time.time() - load_start
    total_items = sum(len(items) for items in kwic_data.values())
    print(f"[AI SUMMARY] Data loaded in {load_time:.2f}s - {total_items} total items")
    
    # Prepare data for AI analysis
    prep_start = time.time()
    print(f"[AI SUMMARY] Preparing data for AI analysis...")
    
    import copy
    clean_data = copy.deepcopy(kwic_data)
    
    # Optionally remove classification and notes
    if remove_annotations:
        for genre_key, items in clean_data.items():
            for item in items:
                item.pop('classification', None)
                item.pop('notes', None)
                item.pop('timestamp', None)
    
    # Convert to the expected format
    formatted_data = {'annotations': {}}
    for genre_key, items in clean_data.items():
        formatted_data['annotations'][genre_key] = {}
        for idx, item in enumerate(items):
            formatted_data['annotations'][genre_key][str(idx)] = item
    
    prep_time = time.time() - prep_start
    print(f"[AI SUMMARY] Data prepared in {prep_time:.2f}s")
    
    try:
        # Import required libraries for AI analysis
        import_start = time.time()
        print(f"[AI SUMMARY] Importing AI libraries...")
        
        from langchain.tools import BaseTool
        from pydantic import BaseModel, Field
        from langchain.chat_models import init_chat_model
        from typing import Optional, Type, Dict, Any
        
        import_time = time.time() - import_start
        print(f"[AI SUMMARY] Libraries imported in {import_time:.2f}s")
        
        # Initialize AWS Bedrock model
        model_id = 'openai.gpt-oss-120b-1:0'
        max_tokens = 128000
        
        model_start = time.time()
        print(f"[AI SUMMARY] Initializing Bedrock model {model_id} with profile {aws_profile}...")
        
        # https://docs.langchain.com/oss/python/langchain/models#initialize-a-model
        model = init_chat_model(
            model_id,
            model_provider="bedrock_converse",
            credentials_profile_name=aws_profile,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        model_time = time.time() - model_start
        print(f"[AI SUMMARY] Model initialized in {model_time:.2f}s")
        
        # Define KWICAnalysisTool Structure
        class KWICAnalysisInput(BaseModel):
            reasoning_level: str = Field(default="high")
            corpus_fullname: str = Field(...)
            corpus_shortname: str = Field(...)
            keyword: str = Field(...)
            random_KWIC_sample: int = Field(default=30)
            kwic_data: Dict[str, Any] = Field(...)
        
        # Stage Bedrock Converse Invoke Statement
        class KWICAnalysisTool(BaseTool):
            name: str = "kwic_analysis"
            description: str = "Analyzes KWIC concordance data"
            args_schema: Type[BaseModel] = KWICAnalysisInput
            model: Any = None
            
            def _run(self, reasoning_level, corpus_fullname, corpus_shortname, 
                    keyword, random_KWIC_sample, kwic_data):
                import json
                from datetime import datetime
                
                metadata = {
                    "corpus_name": corpus_fullname,
                    "corpus_id": corpus_shortname,
                    "keyword": keyword,
                    "reasoning_level": reasoning_level,
                    "timestamp": datetime.now().isoformat()
                }
                
                total_lines = sum(len(items) for items in kwic_data.get('annotations', {}).values())
                
                if random_KWIC_sample > 0 and total_lines > random_KWIC_sample:
                    import random
                    sampled_items = {}
                    all_items = []
                    for genre_id, items in kwic_data['annotations'].items():
                        for idx_str, item in items.items():
                            all_items.append((genre_id, idx_str, item))
                    sampled = random.sample(all_items, random_KWIC_sample)
                    for genre_id, idx_str, item in sampled:
                        if genre_id not in sampled_items:
                            sampled_items[genre_id] = {}
                        sampled_items[genre_id][idx_str] = item
                    kwic_data['annotations'] = sampled_items

                prompt = f"""Analyze:

                            **User Provided Configurations:**
                            - Reasoning level: {reasoning_level}
                            - Corpus: {corpus_fullname} ({corpus_shortname})
                            - Keyword/Phrase: "{keyword}"
                            - Total concordance lines: {total_lines}
                            - Random Sample Size: {random_KWIC_sample}

                            **Corpus Metadata:**
                            {json.dumps(metadata, indent=2)}

                            **KWIC Concordance Data:**
                            {json.dumps(kwic_data, indent=2)}

                            **ANALYSIS TODO:**
                            Question: Given the context (i.e. genre, year, speaker, tone, etc) of the full concordance text data provided, in what *senses* did the speaker mean when they used the keyword '{keyword}'?
                            Step 1. Provide a line by line analysis for the keyword in context
                            Step 2. Provide an analysis of Semantic Prosody
                            Step 3. Provide an analysis of Grammatical Patterns
                            Step 4. Provide an analysis of Evaluation/Comparison
                            Step 5. Provide an analysis of Qualification/Mitigation 

                            **OUTPUT FORMAT GUIDELINES:**
                            1. Do not reference information beyond what is provided in the text. If you are not sure, please indicate uncertainty.
                            2. Provide a disclaimer that the following test is AI generated (model_id = {model_id})
                            3. Please follow the TODO list bulletpointed structure for consistent order of summarized outputs
                        """

                
                conversation = [
                    {"role": "system", "content": "You are a helpful assistant that analyzes concordance line data queried from a corpus."},
                    {"role": "user", "content": prompt}
                ]

                if self.model is None:
                    return "Error: Model not initialized"
                
                # https://docs.langchain.com/oss/python/langchain/models#invoke
                response = model.invoke(conversation)
                #response = self.model.invoke(prompt)


                return response.content if hasattr(response, 'content') else str(response)
            
            async def _arun(self, *args, **kwargs):
                raise NotImplementedError("Async not supported")
        
        # Create and run the tool
        tool_start = time.time()
        print(f"[AI SUMMARY] Creating KWIC analysis tool...")
        
        kwic_tool = KWICAnalysisTool()
        kwic_tool.model = model
        
        print(f"[AI SUMMARY] Running AI analysis (this may take a while)...")
        invoke_start = time.time()
        
        result = kwic_tool._run(
            reasoning_level=reasoning_level,
            corpus_fullname=corpus_fullname,
            corpus_shortname=corpus_shortname,
            keyword=keyword,
            random_KWIC_sample=random_sample,
            kwic_data=formatted_data
        )
        
        invoke_time = time.time() - invoke_start
        tool_time = time.time() - tool_start
        print(f"[AI SUMMARY] AI analysis completed in {invoke_time:.2f}s (total tool time: {tool_time:.2f}s)")
        
        # Parse result if it's a list structure
        parse_start = time.time()
        print(f"[AI SUMMARY] Parsing result...")
        
        reasoning_content = ""
        markdown_content = ""
        
        if isinstance(result, list):
            for item in result:
                if item.get('type') == 'reasoning_content':
                    reasoning_data = item.get('reasoning_content', {})
                    reasoning_content = reasoning_data.get('text', '')
                elif item.get('type') == 'text':
                    markdown_content = item.get('text', '')
        else:
            markdown_content = str(result)
        
        parse_time = time.time() - parse_start
        total_time = time.time() - start_time
        
        print(f"[AI SUMMARY] Result parsed in {parse_time:.2f}s")
        print(f"[AI SUMMARY] TOTAL TIME: {total_time:.2f}s")
        print(f"[AI SUMMARY] Breakdown: Load={load_time:.2f}s, Prep={prep_time:.2f}s, Import={import_time:.2f}s, Model={model_time:.2f}s, Invoke={invoke_time:.2f}s, Parse={parse_time:.2f}s")
        print(f"{'='*80}\n")
        
        return jsonify({
            'success': True,
            'result': markdown_content,
            'reasoning': reasoning_content,
            'metadata': {
                'filename': filename,
                'keyword': keyword,
                'corpus': f"{corpus_fullname} ({corpus_shortname})",
                'reasoning_level': reasoning_level,
                'sample_size': random_sample,
                'temperature': temperature,
                'model': model_id
            }
        })
        
    except Exception as e:
        import traceback
        error_time = time.time() - start_time
        print(f"[AI SUMMARY] ERROR after {error_time:.2f}s: {str(e)}")
        print(f"[AI SUMMARY] Traceback:\n{traceback.format_exc()}")
        print(f"{'='*80}\n")
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/embedding', methods=['POST'])
def api_embedding():
    """API endpoint to run semantic embedding analysis on annotated data."""
    data = request.json
    
    filename = data.get('filename')
    model_name = data.get('model', 'google/embeddinggemma-300m')
    query = data.get('query', '')
    
    if not filename:
        return jsonify({'error': 'No file selected'}), 400
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    # Load annotation file
    annotation_path = ANNOTATIONS_DIR / filename
    if not annotation_path.exists():
        return jsonify({'error': 'Annotation file not found'}), 404
    
    # Get the corresponding KWIC file
    kwic_filename = filename.replace('_annotations.json', '.json')
    if kwic_filename not in get_available_files():
        return jsonify({'error': 'Source KWIC file not found'}), 404
    
    kwic_data = load_kwic_data(kwic_filename)
    annotations = load_annotations(kwic_filename)
    
    if not kwic_data:
        return jsonify({'error': 'Failed to load data'}), 500
    
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        
        # Load the embedding model
        embedding_model = SentenceTransformer(model_name)
        
        # Prepare documents from annotated data
        documents = []
        document_metadata = []
        
        for genre_key, items in kwic_data.items():
            for idx, item in enumerate(items):
                # Check if this item has an annotation
                annotation = annotations.get(genre_key, {}).get(str(idx), {})
                if not annotation:
                    continue
                
                # Use cleaned context if available
                context = item.get('text_clean', item.get('context', ''))
                classification = annotation.get('classification')
                notes = annotation.get('notes', '')
                
                # Format as title | text for better embedding performance
                doc = f"title: {genre_key} | text: {context}"
                documents.append(doc)
                document_metadata.append({
                    'genre': genre_key,
                    'text_id': item.get('text_id', ''),
                    'context': context,
                    'classification': classification,
                    'notes': notes
                })
        
        if not documents:
            return jsonify({'error': 'No annotated data found in this file'}), 400
        
        # Encode query and documents
        query_prompt = f"task: question answering | query: {query}"
        query_embedding = embedding_model.encode(query_prompt, prompt_name="query")
        document_embeddings = embedding_model.encode(documents, prompt_name="document")
        
        # Calculate similarities
        similarities = embedding_model.similarity(query_embedding, document_embeddings)
        
        # Get rankings
        ranked_indices = torch.argsort(similarities, descending=True)[0]
        
        # Prepare results
        all_ranked = []
        for i, idx in enumerate(ranked_indices):
            idx = idx.item()
            all_ranked.append({
                'rank': i + 1,
                'score': float(similarities[0][idx]),
                'genre': document_metadata[idx]['genre'],
                'text_id': document_metadata[idx]['text_id'],
                'context': document_metadata[idx]['context'],
                'classification': document_metadata[idx]['classification'],
                'notes': document_metadata[idx]['notes']
            })
        
        return jsonify({
            'success': True,
            'query': query,
            'model': model_name,
            'total_documents': len(documents),
            'results': all_ranked
        })
        
    except ImportError:
        return jsonify({
            'error': 'sentence-transformers not installed. Run: pip install sentence-transformers torch'
        }), 500
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    print(f"🚀 Starting Annotator App...")
    print(f"📂 Data directory: {DATA_DIR}")
    print(f"📁 Exports directory: {EXPORTS_DIR}")
    print(f"📝 Annotations directory: {ANNOTATIONS_DIR}")
    print(f"📊 Available files: {get_available_files()}")
    app.run(debug=True, host='0.0.0.0', port=5001)