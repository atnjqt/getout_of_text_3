# GetOut-Of-Text3

- Etienne Jacquot

## Getting Started 

Repository to begin exploring corpus lingustics applied to administrative law

- think about corpus of SCOTUS, starting with https://supreme.justia.com/cases/federal/us/volume/
- there are PDFs for volumes 502-605, ranging from 1991 to 2025 present.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- to **download pdf text and html metadata in a database local file** run the following command:

```bash
python supreme_justia.py 502
```

- this script calls the [downloader.py](downloader.py) script to download the PDFs and metadata, and then stores the results in a local SQLite database file named `scotus.db`.
- A sample notebook is provided in [demo.ipynb](demo.ipynb) to explore the data and perform some basic analysis to explore a sampling of a scotus corpus.