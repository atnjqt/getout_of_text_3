# KWIC Annotation Tool

A Flask-based web application for manually coding KWIC (Key Word In Context) concordance lines from corpus linguistics research.

## Features

- 📁 **File Selection**: Dropdown to select from available KWIC JSON files
- 📊 **Progress Tracking**: Visual progress bar and statistics (total, annotated, remaining)
- 🎯 **Classification System**: Code each hit as literal, figurative, neither, or unclear
- 📝 **Notes**: Add optional notes for each annotation
- 💾 **Auto-save**: Annotations saved immediately to JSON files
- 📤 **Export**: Export all annotations with timestamps
- 🎨 **Clean UI**: Modern, responsive interface

## Installation

- Dependencies include: 
  - **web framework**: flask 
  - **natural language processing**: pandas, spacy, scikit-learn,  
  - **embedding models**: sentence-transformers, torch
  - **large language models**: langchain, langchain-aws, 

1. Install dependencies:

  ```bash 
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

- you can install spacy models as needed, e.g.:
  ```bash
  python -m spacy download en_core_web_sm
  ```

## Usage

1. Start the Flask app:
```bash
python3 app.py
```

2. Open your browser to: `http://localhost:5001`

3. Select a KWIC file from the dropdown (e.g., `kwic_coca.json`)

4. Annotate each concordance line:
   - Read the context with highlighted keyword
   - View full text if needed
   - Select classification
   - Add optional notes
   - Click "Save Annotation"

5. Export annotations when done

## File Structure

```
annotator_app/
├── app.py                  # Flask backend
├── templates/
│   └── index.html         # Frontend interface
├── annotations/           # Saved annotations (auto-created)
└── README.md
```

## Data Format

### Input (KWIC JSON):
```json
{
  "genre_key": [
    {
      "text_id": 123,
      "match": "best system",
      "context": "...text with **keyword** highlighted...",
      "full_text": "Complete document text..."
    }
  ]
}
```

### Output (Annotations JSON):
```json
{
  "genre_key": {
    "0": {
      "classification": "literal",
      "notes": "Clear optimal system reference",
      "timestamp": "2025-11-14T10:30:00"
    }
  }
}
```

## API Endpoints

- `GET /` - Main interface
- `GET /api/files` - List available KWIC files
- `GET /api/load/<filename>` - Load KWIC data and existing annotations
- `POST /api/save` - Save annotation
- `GET /api/export/<filename>` - Export annotations

## Notes

- Annotations are saved to `annotations/` directory
- Each file's annotations stored separately
- Progress is tracked across sessions
- Annotations include timestamps for auditing
