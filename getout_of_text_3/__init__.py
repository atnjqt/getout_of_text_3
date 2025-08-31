from getout_of_text_3._config import options

# Expose main corpus class
from getout_of_text_3.corpus import LegalCorpus

# ...existing code...
#from getout_of_text_3.io.file import _read_file as read_file
#from getout_of_text_3.tools import sjoin, sjoin_nearest

#import getout_of_text_3.datasets

# make the interactive namespace easier to use
# for `from getout_of_text3 import *` demos.
import getout_of_text_3 as got3
import pandas as pd
import numpy as np

# Move version import to the end to avoid circular import issues
from . import _version
__version__ = _version.get_versions()["version"]