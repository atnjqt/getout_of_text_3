import pandas as pd
import os

class LegalCorpus:
    """
    Main class for handling legal corpora and BYU datasets.
    """
    def __init__(self, data_dir):
        """
        Initialize with the directory containing BYU data files.
        """
        self.data_dir = data_dir

    def list_files(self):
        """
        List all files in the data directory.
        """
        return [f for f in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, f))]

    def read_byu_file(self, filename, **kwargs):
        """
        Read a BYU data file into a pandas DataFrame.
        Supports CSV and TSV formats.
        """
        file_path = os.path.join(self.data_dir, filename)
        if filename.endswith('.csv'):
            return pd.read_csv(file_path, **kwargs)
        elif filename.endswith('.tsv'):
            return pd.read_csv(file_path, sep='\t', **kwargs)
        else:
            raise ValueError("Unsupported file format. Please use CSV or TSV.")

    # Placeholder for future KWIC, frequency, concordance methods
    # def kwic(self, ...):
    #     pass
    # def keyword_frequencies(self, ...):
    #     pass
    # def collocates(self, ...):
    #     pass
    # def concordance(self, ...):
    #     pass
