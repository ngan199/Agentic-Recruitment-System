import os
import unittest
from agents.cv_parser import extract_file

class TestExtractors(unittest.TestCase):
    def setUp(self):
        self.test_files = {
            'pdf': 'tests/developer_cv.pdf',
            'docx': 'tests/developer_cv.docx',
            'pptx': 'tests/developer_cv.pptx'
        }

    def test_extract_pdf(self):
        result = extract_file(self.test_files['pdf'])
        self.assertIn('text', result)
        self.assertGreater(len(result['text']), 0)

    def test_extract_docx(self):
        result = extract_file(self.test_files['docx'])
        self.assertIn('paragraphs', result)
        self.assertGreater(len(result['paragraphs']), 0)

    def test_extract_pptx(self):
        result = extract_file(self.test_files['pptx'])
        self.assertIn('slides', result)
        self.assertGreater(len(result['slides']), 0)

if __name__ == '__main__':
    unittest.main()