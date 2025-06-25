# GetOut-Of-Text3

- Etienne Jacquot

## Getting Started 

Repository to begin exploring corpus lingustics applied to administrative law

- think about corpus of SCOTUS, starting with https://supreme.justia.com/cases/federal/us/603/22-451/

- really simple bootstrapped AI vibe stuff here:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# download a PDF URL
python download_pdf.py

# view db 
python display_db.py
```