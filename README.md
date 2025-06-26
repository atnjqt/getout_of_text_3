# GetOut-Of-Text3

- Etienne Jacquot

## Getting Started 

Repository to begin exploring corpus lingustics applied to administrative law

- think about corpus of SCOTUS, starting with https://supreme.justia.com/cases/federal/us/volume/
- their terms of service allow for downloading data, please see here for reference https://www.justia.com/terms-of-service/ 
    > **6. MEMBER CONDUCT:**
    > - *h. Forge headers or otherwise manipulate identifiers or other data in order to disguise the origin of any Content transmitted through the Service or to manipulate your presence on our sites;* and 
    > - *m. Take any action that imposes an unreasonably or disproportionately large load on our infrastructure.*
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

- run for an automated loop of numbers (i.e. `505-510`) pass `db_name` as the 2nd arg and `auto` as the 3rd argument:
    ```bash
    for i in {505..510}; do python supreme_justia.py $i pdf_texts.pdf auto; done
    ```

## ETL

- A sample notebook is provided in [demo.ipynb](demo.ipynb) to explore the data and perform some basic analysis to explore a sampling of a scotus corpus by reading the DB into a pandas dataframe.

- in a scotus document, usually sections of text are described as 'Opinion of the Court', 'Opinion of JUSTICE NAME', or 'JUSTCE NAME, dissenting'. There are likely steps to chunk the text into who's opinion it was related to. I think over time we could see how the opinions change by using corpus stuff.
