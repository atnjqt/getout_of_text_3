#!/usr/bin/env python3
"""
Test suite for enhanced getout_of_text_3 functionality
"""

import unittest
import tempfile
import os
import getout_of_text_3 as got3

class TestEnhancedFunctionality(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        # Create temporary directory with sample data
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample COCA files
        sample_texts = {
            'acad': [
                'This is a legal academic text about constitutional law.',
                'The legal system requires constitutional interpretation.',
                'Legal scholars study constitutional principles.'
            ],
            'news': [
                'Breaking news: legal challenges to new legislation.',
                'The court ruled on constitutional grounds.',
                'Legal experts analyze the constitutional implications.'
            ]
        }
        
        for genre, texts in sample_texts.items():
            file_path = os.path.join(self.temp_dir, f'text_{genre}.txt')
            with open(file_path, 'w', encoding='utf-8') as f:
                for text in texts:
                    f.write(text + '\n')
    
    def tearDown(self):
        """Clean up test data"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_read_corpora(self):
        """Test corpus reading functionality"""
        corpus_data = got3.read_corpora(self.temp_dir, "test_corpus", ['acad', 'news'])
        
        self.assertIn('acad', corpus_data)
        self.assertIn('news', corpus_data)
        self.assertEqual(len(corpus_data['acad']), 3)
        self.assertEqual(len(corpus_data['news']), 3)
    
    def test_search_keyword_corpus(self):
        """Test keyword search functionality"""
        corpus_data = got3.read_corpora(self.temp_dir, "test_corpus", ['acad', 'news'])
        
        results = got3.search_keyword_corpus(
            "legal", 
            corpus_data, 
            case_sensitive=False, 
            show_context=False
        )
        
        self.assertIn('acad', results)
        self.assertIn('news', results)
        # Should find legal mentions in both genres
        self.assertTrue(len(results['acad']) > 0)
        self.assertTrue(len(results['news']) > 0)
    
    def test_keyword_frequency_analysis(self):
        """Test frequency analysis"""
        corpus_data = got3.read_corpora(self.temp_dir, "test_corpus", ['acad', 'news'])
        
        freq_results = got3.keyword_frequency_analysis("legal", corpus_data)
        
        self.assertIn('acad', freq_results)
        self.assertIn('news', freq_results)
        self.assertIn('count', freq_results['acad'])
        self.assertIn('freq_per_1000', freq_results['acad'])
        
        # Should find at least some occurrences
        self.assertGreater(freq_results['acad']['count'], 0)
        self.assertGreater(freq_results['news']['count'], 0)
    
    def test_find_collocates(self):
        """Test collocate analysis"""
        corpus_data = got3.read_corpora(self.temp_dir, "test_corpus", ['acad', 'news'])
        
        collocate_results = got3.find_collocates(
            "legal", 
            corpus_data, 
            window_size=3, 
            min_freq=1
        )
        
        self.assertIn('all_collocates', collocate_results)
        self.assertIn('by_genre', collocate_results)
        self.assertTrue(len(collocate_results['all_collocates']) > 0)
    
    def test_legal_corpus_class(self):
        """Test LegalCorpus class functionality"""
        corpus = got3.LegalCorpus()
        
        # Test corpus loading
        corpus_data = corpus.read_corpora(self.temp_dir, "test_class_corpus", ['acad', 'news'])
        
        # Test corpus management
        self.assertIn("test_class_corpus", corpus.list_corpora())
        retrieved_corpus = corpus.get_corpus("test_class_corpus")
        self.assertEqual(len(retrieved_corpus), 2)  # 2 genres
        
        # Test search methods
        search_results = corpus.search_keyword_corpus("constitutional", corpus_data)
        freq_results = corpus.keyword_frequency_analysis("constitutional", corpus_data)
        collocate_results = corpus.find_collocates("constitutional", corpus_data)
        
        # Basic validation
        self.assertIsInstance(search_results, dict)
        self.assertIsInstance(freq_results, dict)
        self.assertIsInstance(collocate_results, dict)

if __name__ == '__main__':
    print("Running enhanced getout_of_text_3 test suite...")
    unittest.main(verbosity=2)
