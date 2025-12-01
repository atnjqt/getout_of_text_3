import pandas as pd
import os
import re
import multiprocessing as mp
from collections import defaultdict, Counter
from functools import partial
try:
    from tqdm import tqdm  # progress bar for long corpus builds
except ImportError:  # graceful fallback if tqdm not installed
    def tqdm(iterable, **kwargs):
        return iterable
try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("⚠️ NLTK not available. Some features may use fallback methods.")

class LegalCorpus:

    @staticmethod
    def _process_collocates(args):
        genre_year, df, keyword, window_size, min_freq, case_sensitive = args
        # Use loose substring regex for matching
        flags = 0 if case_sensitive else re.IGNORECASE
        pattern = re.compile(re.escape(keyword), flags)
        genre_counter = Counter()
        keyword_instances = 0
        for text in df['text']:
            text_str = str(text)
            # Tokenize (use split for speed, fallback to nltk if available and desired)
            words = text_str.split()
            # Find all positions where the keyword appears as a substring (loose match)
            positions = [i for i, w in enumerate(words) if pattern.search(w)]
            keyword_instances += len(positions)
            for pos in positions:
                start = max(0, pos - window_size)
                end = min(len(words), pos + window_size + 1)
                context_words = words[start:pos] + words[pos+1:end]
                context_words = [w for w in context_words if w.isalpha() and len(w) > 2]
                genre_counter.update(context_words)
        return genre_year, genre_counter, keyword_instances

    @staticmethod
    def _process_genre(args):
        genre, year_dict, keyword, flags, relative = args
        pattern = re.compile(re.escape(keyword), flags)
        genre_count = 0
        genre_tokens = 0
        years = {}
        per_text_local = {}
        for year, df in year_dict.items():
            year_count = 0
            year_tokens = 0
            for idx, row in df.iterrows():
                text_id = str(row['text_id']) if 'text_id' in row else str(idx)
                s = str(row['text'])
                count = len(pattern.findall(s))
                tokens = len(s.split())
                genre_count += count
                genre_tokens += tokens
                year_count += count
                year_tokens += tokens
                key = f"{genre}_{year}_{text_id}"
                per_text_local[key] = {'count': count, 'tokens': tokens}
            years[year] = {'count': year_count, 'tokens': year_tokens}
        entry = {'genre': genre, 'count': genre_count, 'tokens': genre_tokens, 'years': years}
        if relative and genre_tokens:
            entry['rel_per_10k'] = (genre_count / genre_tokens) * 10000
        return entry, per_text_local, genre_count, genre_tokens
    """
    Main class for handling legal corpora and BYU datasets.
    
    This class provides comprehensive functionality for working with COCA (Corpus of Contemporary 
    American English) and other legal text corpora, designed specifically for legal scholars 
    and researchers working on open science projects.
    """
    
    def __init__(self, data_dir=None):
        """
        Initialize with the directory containing BYU data files.
        
        Parameters:
        - data_dir: Directory containing corpus files (optional)
        """
        self.data_dir = data_dir
        self.corpora = {}
        self._ensure_nltk_data()

    def _ensure_nltk_data(self):
        """Ensure required NLTK data is available."""
        if NLTK_AVAILABLE:
            try:
                nltk.download('punkt_tab', quiet=True)
                nltk.download('stopwords', quiet=True)
            except:
                print("⚠️ Could not download NLTK data. Some features may be limited.")

    def list_files(self):
        """
        List all files in the data directory.
        """
        if not self.data_dir:
            raise ValueError("No data directory specified. Set data_dir first.")
        return [f for f in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, f))]

    def read_byu_file(self, filename, **kwargs):
        """
        Read a BYU data file into a pandas DataFrame.
        Supports CSV and TSV formats.
        """
        if not self.data_dir:
            raise ValueError("No data directory specified. Set data_dir first.")
            
        file_path = os.path.join(self.data_dir, filename)
        if filename.endswith('.csv'):
            return pd.read_csv(file_path, **kwargs)
        elif filename.endswith('.tsv'):
            return pd.read_csv(file_path, sep='\t', **kwargs)
        else:
            raise ValueError("Unsupported file format. Please use CSV or TSV.")

    def read_corpus(self, dir_of_text_files=None, corpus_name='coca', show_progress=True, show_file_progress=True, log_every=0):
        """Build a structured dictionary from corpus text folders.

        Structure returned depends on corpus_name (both use nested structure):
        - 'coca'/'diy': {genre: {year_or_id: DataFrame(['text_id', 'text'])}}
        - 'glowbe': {country_code: {file_id: DataFrame(['text_id', 'text'])}}

        Parameters:
            dir_of_text_files (str|None): Root directory containing corpus subfolders (defaults to self.data_dir)
            corpus_name (str): Corpus type - 'coca' (default), 'glowbe', or 'diy' (uses COCA format)
            show_progress (bool): Show top-level progress bar.
            show_file_progress (bool): Show per-file progress bar (requires tqdm).
            log_every (int): If > 0, print a running line-count every N captured lines.

        Notes on progress behavior:
            Previously the single progress bar hit 100% once the last genre started, while large
            final-genre files were still being parsed, creating the appearance of a "hang".
            The added per-file progress bar (and optional line logging) provides visibility during
            that final stretch.
        """
        if dir_of_text_files is None:
            if not self.data_dir:
                raise ValueError("No data directory specified. Set data_dir first.")
            dir_of_text_files = self.data_dir

        # Route to appropriate parser based on corpus_name
        if corpus_name in ['coca', 'diy']:
            return self._read_coca_structure(dir_of_text_files, show_progress, show_file_progress, log_every)
        elif corpus_name == 'glowbe':
            return self._read_glowbe_structure(dir_of_text_files, show_progress, show_file_progress, log_every)
        else:
            raise ValueError(f"Unknown corpus_name: '{corpus_name}'. Must be 'coca', 'glowbe', or 'diy'.")

    def _read_coca_structure(self, dir_of_text_files, show_progress, show_file_progress, log_every):
        """Parse COCA corpus structure: text_GENRE_YEAR or text_GENRE_ID format.
        
        Returns: {genre: {year_or_id: DataFrame}}
        """
        coca_dict = {}
        genre_folders = [f for f in os.listdir(dir_of_text_files) if f.startswith('text_')]

        genre_iter = tqdm(genre_folders, desc="COCA Genres", unit="genre") if show_progress else genre_folders

        for genre_folder in genre_iter:
            genre = genre_folder.split('_')[1]
            print(f"Processing genre: {genre}")
            genre_path = os.path.join(dir_of_text_files, genre_folder)

            # Gather candidate files
            genre_files = [fn for fn in os.listdir(genre_path) if fn.startswith('text_') and fn.endswith('.txt')]
            file_iter = tqdm(genre_files, desc=f"{genre} files", unit="file", leave=False) if (show_file_progress and show_progress) else genre_files

            genre_dict = {}
            for filename in file_iter:
                year_match = re.search(r'_(\d{4})\.txt$', filename)
                file_num_match = None
                if not year_match and genre in ['web', 'blog']:
                    file_num_match = re.search(r'_(\d+)\.txt$', filename)

                file_path = os.path.join(genre_path, filename)
                text_rows = []
                captured_lines = 0
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            if not line.startswith('@@'):
                                continue
                            line = line.strip()
                            parts = line.split(' ', 1)
                            if len(parts) != 2:
                                continue
                            id_part = parts[0][2:]  # e.g., '@@12345' -> '12345'
                            text_part = parts[1]
                            text_rows.append({'text_id': id_part, 'text': text_part})
                            captured_lines += 1
                            if log_every and captured_lines % log_every == 0:
                                print(f"  {genre}: {captured_lines} lines captured so far...")
                except Exception as e:
                    print(f"  ⚠️ Failed reading {file_path}: {e}")

                # Assign DataFrame to correct year/file_num
                if text_rows:
                    if year_match:
                        year = str(year_match.group(1))
                        genre_dict[year] = pd.DataFrame(text_rows)
                    elif file_num_match:
                        file_num = file_num_match.group(1)
                        genre_dict[file_num] = pd.DataFrame(text_rows)
            print(f"Finished genre: {genre} (total files: {len(genre_dict)})")
            coca_dict[genre] = genre_dict

        return coca_dict

    def _read_glowbe_structure(self, dir_of_text_files, show_progress, show_file_progress, log_every):
        """Parse GloWbE corpus structure: text_COUNTRYCODE_* folders with w_*.txt files.
        
        Returns nested structure like COCA: {country_code: {file_id: DataFrame}}
        """
        glowbe_dict = {}
        folders = [f for f in os.listdir(dir_of_text_files) if f.startswith('text_')]

        folder_iter = tqdm(folders, desc="GloWbE Countries", unit="country") if show_progress else folders

        for folder in folder_iter:
            parts = folder.split('_')
            # Extract country code (2nd part after 'text_')
            # e.g., text_us_genl_ksl -> country='us'
            country = parts[1] if len(parts) > 1 else 'unknown'
            
            folder_path = os.path.join(dir_of_text_files, folder)
            
            # GloWbE files use w_*.txt naming (e.g., w_us_g19.txt)
            files = [fn for fn in os.listdir(folder_path) if fn.startswith('w_') and fn.endswith('.txt')]
            file_iter = tqdm(files, desc=f"{country} files", unit="file", leave=False) if (show_file_progress and show_progress) else files
            
            # Initialize country dict if needed
            if country not in glowbe_dict:
                glowbe_dict[country] = {}
            
            for filename in file_iter:
                # Extract file ID from filename: w_us_g19.txt -> 'g19'
                file_id_match = re.search(r'w_[a-z]{2}_([a-z0-9]+)\.txt$', filename)
                if not file_id_match:
                    print(f"  ⚠️ Could not parse file ID from {filename}, skipping...")
                    continue
                file_id = file_id_match.group(1)
                
                file_path = os.path.join(folder_path, filename)
                text_rows = []
                captured_lines = 0
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            # GloWbE uses ## as line marker (not @@)
                            if not line.startswith('##'):
                                continue
                            line = line.strip()
                            parts_line = line.split(' ', 1)
                            if len(parts_line) != 2:
                                continue
                            id_part = parts_line[0][2:]  # e.g., '##12345' -> '12345'
                            text_part = parts_line[1]
                            text_rows.append({'text_id': id_part, 'text': text_part})
                            captured_lines += 1
                            if log_every and captured_lines % log_every == 0:
                                print(f"  {country}/{file_id}: {captured_lines} lines captured so far...")
                except Exception as e:
                    print(f"  ⚠️ Failed reading {file_path}: {e}")
                
                # Store DataFrame for this file_id
                if text_rows:
                    glowbe_dict[country][file_id] = pd.DataFrame(text_rows)
            
            print(f"Finished country: {country} (total files: {len(glowbe_dict[country])})")
        
        return glowbe_dict

    def _search_single_genre(self, genre_year_df_keyword_args):
        """
        Helper function for parallel processing - searches a single genre.
        
        Args:
            genre_year_df_keyword_args: Tuple of (genre_year, df, keyword, case_sensitive, 
                                                  show_context, context_words)
        
        Returns:
            Tuple of (genre_year, results_list, genre_hits)
        """
        genre_year, df, keyword, case_sensitive, show_context, context_words = genre_year_df_keyword_args
        
        # Prepare search pattern
        if case_sensitive:
            pattern = re.compile(r'\b' + re.escape(keyword) + r'\b')
        else:
            pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
        
        results = []
        genre_hits = 0
        
        for idx, text in df['text'].items():
            text_str = str(text)
            matches = pattern.findall(text_str)
            if matches:
                genre_hits += len(matches)
                if show_context:
                    for match in pattern.finditer(text_str):
                        start, end = match.span()
                        words_before_match = text_str[:start].split()
                        words_after_match = text_str[end:].split()
                        context_before = ' '.join(words_before_match[-context_words:]) if words_before_match else ""
                        matched_word = text_str[start:end]
                        context_after = ' '.join(words_after_match[:context_words]) if words_after_match else ""
                        context_display = f"{context_before} **{matched_word}** {context_after}".strip()
                        results.append({
                            'text_id': idx,
                            'match': matched_word,
                            'context': context_display,
                            'full_text': text_str[:100] + "..." if len(text_str) > 100 else text_str
                        })
                else:
                    results.append({
                        'text_id': idx,
                        'matches': len(matches),
                        'full_text': text_str[:100] + "..." if len(text_str) > 100 else text_str
                    })
        
        return genre_year, results, genre_hits

    def _flatten_corpus_structure(self, db_dict):
        """
        Helper method to flatten nested corpus structure to work with existing search logic.
        
        Input formats supported:
        - Flat: {genre_year: DataFrame} -> returns as-is
        - Nested: {genre: {year: DataFrame}} -> flattens to {genre_year: DataFrame}
        
        Returns:
        - Flat dictionary: {genre_year: DataFrame}
        """
        # Check if this is already a flat structure
        # Look at first value to determine structure
        if not db_dict:
            return {}
            
        first_key = next(iter(db_dict))
        first_value = db_dict[first_key]
        
        # If first value is a DataFrame, assume flat structure
        if isinstance(first_value, pd.DataFrame):
            return db_dict
            
        # If first value is a dict, assume nested structure and flatten
        elif isinstance(first_value, dict):
            flat_dict = {}
            for genre, years_dict in db_dict.items():
                for year, df in years_dict.items():
                    if isinstance(df, pd.DataFrame):
                        flat_key = f"{genre}_{year}"
                        flat_dict[flat_key] = df
                    else:
                        print(f"Skipping {genre}/{year}: not a DataFrame (type={type(df)})")
            return flat_dict
        else:
            # Unknown structure, return as-is and let it fail gracefully
            print(f"Warning: Unknown corpus structure type for key '{first_key}': {type(first_value)}")
            return db_dict
    
    def read_corpora(self, dir_of_text_files, corpora_name, genre_list=None):
        """
        Read COCA corpus files from a directory and organize by genre.
        
        Parameters:
        - dir_of_text_files: Directory containing the text files
        - corpora_name: Name identifier for this corpus collection
        - genre_list: List of genres to process (default: COCA standard genres)
        
        Returns:
        - Dictionary with genre keys and DataFrames as values
        """
        if genre_list is None:
            genre_list = ['acad', 'blog', 'fic', 'mag', 'news', 'spok', 'tvm', 'web']
        
        print(f"📚 Loading {corpora_name} corpus from {dir_of_text_files}")
        print("=" * 60)
        
        corpus_data = {}
        
        for genre in genre_list:
            print(f"📂 Processing {genre}...")
            
            try:
                # Look for both db_ and text_ prefixed files
                for prefix in ['db_', 'text_', '']:
                    file_pattern = f"{prefix}{genre}.txt"
                    file_path = os.path.join(dir_of_text_files, file_pattern)
                    
                    if os.path.exists(file_path):
                        corpus_data[genre] = pd.read_csv(
                            file_path,
                            sep="\t",
                            header=None,
                            names=["text"],
                            on_bad_lines='skip',
                            quoting=3
                        )
                        print(f"  ✅ {file_pattern}: {corpus_data[genre].shape}")
                        break
                else:
                    print(f"  ❌ No file found for {genre}")
                    
            except Exception as e:
                print(f"  ❌ Error reading {genre}: {e}")
        
        # Store in the corpus collection
        self.corpora[corpora_name] = corpus_data
        
        print(f"\n🎯 SUMMARY:")
        print(f"   - {corpora_name}: {len(corpus_data)} genres loaded")
        print(f"   - Total corpora in collection: {len(self.corpora)}")
        
        return corpus_data

    def search_keyword_corpus(self, keyword, db_dict, case_sensitive=False, show_context=True, context_words=5, output='print', parallel=True, n_jobs=None):
        """
        Search for a keyword across a corpus dict.
        
        Supports both structures:
        - Flat: {genre_year: DataFrame} (legacy format)
        - Nested: {genre: {year: DataFrame}} (new full COCA format)
        
        Parameters:
        - keyword: The word/phrase to search for
        - db_dict: Dictionary structure containing DataFrames
        - case_sensitive: Whether to perform case-sensitive search
        - show_context: Whether to show surrounding context
        - context_words: Number of words to show on each side for context
        - output: 'print' to display results, 'json' to return structured data
        - parallel: Whether to use parallel processing (default: True)
        - n_jobs: Number of parallel processes (default: CPU count - 1)
        
        Returns:
        - Dictionary with search results
        """
        if output == 'print':
            print(f"🔍 COCA Corpus Search: '{keyword}'")
            print("=" * 60)

        results = defaultdict(list)
        total_hits = 0

        # Detect structure type and flatten if needed
        flat_dict = self._flatten_corpus_structure(db_dict)
        
        if parallel and len(flat_dict) > 1:
            # Use parallel processing for multiple genres
            if n_jobs is None:
                n_jobs = max(1, mp.cpu_count() - 1)  # Leave one CPU free
            
            if output == 'print':
                print(f"🚀 Using parallel processing with {n_jobs} processes...")
            
            # Prepare arguments for parallel processing
            args_list = [
                (genre_year, df, keyword, case_sensitive, show_context, context_words)
                for genre_year, df in flat_dict.items()
            ]
            
            # Process in parallel
            try:
                with mp.Pool(processes=n_jobs) as pool:
                    parallel_results = pool.map(self._search_single_genre, args_list)
                
                # Collect results
                for genre_year, genre_results, genre_hits in parallel_results:
                    results[genre_year] = genre_results
                    total_hits += genre_hits
                    
                    if output == 'print':
                        print(f"\n📚 {genre_year.upper()} :")
                        print("-" * 30)
                        if genre_hits > 0:
                            if show_context:
                                for result in genre_results:
                                    print(f"  📝 Text {result['text_id']}: {result['context']}")
                            print(f"  ✅ Found {genre_hits} occurrence(s) in {genre_year}")
                        else:
                            print(f"  ❌ No matches found in {genre_year}")
                            
            except Exception as e:
                print(f"⚠️ Parallel processing failed: {e}. Falling back to sequential processing...")
                parallel = False
        
        if not parallel or len(flat_dict) <= 1:
            # Sequential processing (fallback or single genre)
            for genre_year, df in flat_dict.items():
                _, genre_results, genre_hits = self._search_single_genre(
                    (genre_year, df, keyword, case_sensitive, show_context, context_words)
                )
                results[genre_year] = genre_results
                total_hits += genre_hits
                
                if output == 'print':
                    print(f"\n� {genre_year.upper()} :")
                    print("-" * 30)
                    if genre_hits > 0:
                        if show_context:
                            for result in genre_results:
                                print(f"  📝 Text {result['text_id']}: {result['context']}")
                        print(f"  ✅ Found {genre_hits} occurrence(s) in {genre_year}")
                    else:
                        print(f"  ❌ No matches found in {genre_year}")
        
        if output == 'print':
            print(f"\n🎯 SUMMARY:")
            print(f"Total hits across all genre_years: {total_hits}")
            print(f"Genre_years with matches: {len([g for g in results if results[g]])}")
            return dict(results)
        elif output == 'json':
            json_results = {}
            for genre_year, items in results.items():
                genre_dict = {}
                # Track occurrence count per text_id to create unique keys
                text_id_counters = {}
                
                for item in items:
                    text_id = str(item['text_id'])
                    
                    # Increment counter for this text_id
                    if text_id not in text_id_counters:
                        text_id_counters[text_id] = 0
                    text_id_counters[text_id] += 1
                    
                    # Create unique key with occurrence number
                    unique_key = f"{text_id}_{text_id_counters[text_id]}"
                    
                    # Store the match with unique key
                    if 'context' in item:
                        genre_dict[unique_key] = item['context']
                    else:
                        genre_dict[unique_key] = f"{item['matches']} matches"
                
                json_results[genre_year] = genre_dict
            return json_results

    def find_collocates(self, keyword, db_dict, window_size=5, min_freq=2, case_sensitive=False, parallel=True, n_jobs=None):
        """
        Find words that frequently appear near the keyword (collocates).
        
        Supports both corpus structures:
        - Flat: {genre_year: DataFrame} (legacy format)
        - Nested: {genre: {year: DataFrame}} (new full COCA format)
        
        Parameters:
        - keyword: Target word to find collocates for
        - db_dict: Dictionary structure containing DataFrames
        - window_size: Number of words to look at on each side
        - min_freq: Minimum frequency for a word to be considered a collocate
        - case_sensitive: Whether to perform case-sensitive search
        - parallel: Use multiprocessing by genre (default True)
        - n_jobs: Number of processes (default n-1)
        
        Returns:
        - Dictionary with collocate data
        """
        print(f"🔗 Collocate Analysis for '{keyword}' (window: ±{window_size} words, loose substring match)")
        print("=" * 60)

        # Flatten structure if needed
        flat_dict = self._flatten_corpus_structure(db_dict)
        all_collocates = Counter()
        genre_collocates = {}
        genre_results = []

        items = list(flat_dict.items())
        if parallel and len(items) > 1:
            if n_jobs is None:
                n_jobs = max(1, mp.cpu_count() - 1)
            try:
                args_list = [(genre_year, df, keyword, window_size, min_freq, case_sensitive) for genre_year, df in items]
                with mp.Pool(processes=n_jobs) as pool:
                    results = pool.map(LegalCorpus._process_collocates, args_list)
                for genre_year, genre_counter, keyword_instances in results:
                    genre_collocates[genre_year] = genre_counter
                    all_collocates.update(genre_counter)
                    genre_results.append((genre_year, genre_counter, keyword_instances))
            except Exception as e:
                print(f"⚠️ Parallel processing failed: {e}. Falling back to sequential processing...")
                parallel = False

        if not parallel or len(items) <= 1:
            for genre_year, df in items:
                genre_year, genre_counter, keyword_instances = LegalCorpus._process_collocates((genre_year, df, keyword, window_size, min_freq, case_sensitive))
                genre_collocates[genre_year] = genre_counter
                all_collocates.update(genre_counter)
                genre_results.append((genre_year, genre_counter, keyword_instances))

        # Print top collocates for each genre
        for genre_year, genre_counter, keyword_instances in genre_results:
            print(f"\n📚 {genre_year.upper()} Genre Collocates:")
            top_collocates = genre_counter.most_common(10)
            if top_collocates:
                print(f"  Found {keyword_instances} instances of '{keyword}' in {genre_year}")
                for word, freq in top_collocates:
                    marker = "  " if freq >= min_freq else "* "
                    print(f"{marker}{word:15s}: {freq:3d} times")
            else:
                print(f"  Found {keyword_instances} instances, but no significant collocates")

        print(f"\n🎯 TOP OVERALL COLLOCATES (min frequency: {min_freq}):")
        print("-" * 40)
        top_overall = all_collocates.most_common(20)
        for word, freq in top_overall:
            if freq >= min_freq:
                print(f"{word:15s}: {freq:3d} occurrences")

        return {
            'all_collocates': dict(all_collocates),
            'by_genre': dict(genre_collocates),
        }

    def get_corpus(self, corpora_name):
        """
        Get a previously loaded corpus by name.
        
        Parameters:
        - corpora_name: Name of the corpus to retrieve
        
        Returns:
        - Dictionary of DataFrames for the requested corpus
        """
        if corpora_name not in self.corpora:
            raise ValueError(f"Corpus '{corpora_name}' not found. Available: {list(self.corpora.keys())}")
        return self.corpora[corpora_name]

    def list_corpora(self):
        """
        List all loaded corpora.
        
        Returns:
        - List of corpus names
        """
        return list(self.corpora.keys())

    def corpus_summary(self):
        """
        Display a summary of all loaded corpora.
        """
        print("📚 CORPUS COLLECTION SUMMARY")
        print("=" * 50)
        
        if not self.corpora:
            print("No corpora loaded.")
            return
            
        for name, corpus in self.corpora.items():
            print(f"\n🔍 {name}:")
            for genre, df in corpus.items():
                total_texts = len(df)
                total_words = sum(len(str(text).split()) for text in df['text'])
                print(f"  {genre:8s}: {total_texts:6d} texts, ~{total_words:8d} words")

    # Legacy method for backward compatibility
    def kwic(self, keyword, db_dict, **kwargs):
        """Legacy method - use search_keyword_corpus instead."""
        print("⚠️ kwic() is deprecated. Use search_keyword_corpus() instead.")
        return self.search_keyword_corpus(keyword, db_dict, **kwargs)

    def keyword_frequency_analysis(self, keyword, db_dict, case_sensitive=False, relative=True, parallel=True, n_jobs=None):
        """Compute frequency of a keyword across a nested corpus dict (genre -> year -> DataFrame),
        with per-genre, per-year, and per-text_id breakdowns. Now supports parallel processing by genre.

        Parameters:
            keyword (str): Term to count
            db_dict (dict): genre -> year -> DataFrame(['text_id', 'text'])
            case_sensitive (bool): case sensitivity flag
            relative (bool): include per 10k tokens metric
            parallel (bool): use multiprocessing (default True)
            n_jobs (int|None): number of processes (default: n-1)
        Returns:
            dict summary
        """
        if not keyword:
            raise ValueError("keyword must be a non-empty string")
        flags = 0 if case_sensitive else re.IGNORECASE
        genres = list(db_dict.items())
        results_list = []
        total_count = 0
        grand_tokens = 0
        per_text = {}

        if parallel and len(genres) > 1:
            if n_jobs is None:
                n_jobs = max(1, mp.cpu_count() - 1)
            try:
                args_list = [(genre, year_dict, keyword, flags, relative) for genre, year_dict in genres]
                with mp.Pool(processes=n_jobs) as pool:
                    results = pool.map(LegalCorpus._process_genre, args_list)
                for entry, per_text_local, genre_count, genre_tokens in results:
                    results_list.append(entry)
                    per_text.update(per_text_local)
                    total_count += genre_count
                    grand_tokens += genre_tokens
            except Exception as e:
                print(f"⚠️ Parallel processing failed: {e}. Falling back to sequential processing...")
                parallel = False

        if not parallel or len(genres) <= 1:
            for genre, year_dict in genres:
                entry, per_text_local, genre_count, genre_tokens = LegalCorpus._process_genre((genre, year_dict, keyword, flags, relative))
                results_list.append(entry)
                per_text.update(per_text_local)
                total_count += genre_count
                grand_tokens += genre_tokens

        results_list.sort(key=lambda x: x['count'], reverse=True)

        summary = {
            'keyword': keyword,
            'total_count': total_count,
            'by_genre': results_list,
            'grand_total_tokens': grand_tokens,
            'per_text': per_text
        }
        print(f"\U0001f4ca Frequency Analysis for '{keyword}' (case_sensitive={case_sensitive}, loose substring match)")
        print("=" * 60)
        for r in results_list:
            if relative and 'rel_per_10k' in r:
                print(f"  {r['genre']:8s}: {r['count']:6d} hits | {r['tokens']:8d} tokens | {r['rel_per_10k']:.2f} /10k")
            else:
                print(f"  {r['genre']:8s}: {r['count']:6d} hits | {r['tokens']:8d} tokens")
        print("-" * 60)
        print(f"TOTAL: {total_count} hits across {len(results_list)} genres (~{grand_tokens} tokens)")
        return summary
