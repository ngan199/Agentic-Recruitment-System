import unittest
import os
import json

# Ensure project root is in path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.cv_parser.text_extractor import (
    extract_file,
    extract_pdf,
    extract_docx,
    extract_pptx,
)


class TestExtractors(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_files = {
            'pdf': "tests/test_cv.pdf",
            'docx': "tests/test_cv.docx",
            'pptx': "tests/test_cv.pptx",
        }

        for f in cls.test_files.values():
            assert os.path.exists(f), f"Missing test file: {f}"

    # ---- PDF Tests ----
    def test_extract_pdf_structured(self):
        result = extract_pdf(self.test_files['pdf'])
        self.assertIn("pages", result)
        self.assertGreater(len(result["pages"]), 0)
        self.assertTrue(
            any(p.get("text") or p.get("ocr_text") for p in result["pages"])
        )

    def test_extract_pdf_text_only(self):
        result = extract_file(self.test_files['pdf'], prefer_text=True)
        self.assertIn("text", result)
        self.assertGreater(len(result["text"]), 0)

    # ---- DOCX Tests ----
    def test_extract_docx_structured(self):
        result = extract_docx(self.test_files['docx'])
        self.assertIn("paragraphs", result)
        self.assertGreater(len(result["paragraphs"]), 0)

    def test_extract_docx_text_only(self):
        result = extract_file(self.test_files['docx'], prefer_text=True)
        self.assertIn("text", result)
        self.assertGreater(len(result["text"]), 0)

    # ---- PPTX Tests ----
    def test_extract_pptx_structured(self):
        result = extract_pptx(self.test_files['pptx'])
        self.assertIn("slides", result)
        self.assertGreater(len(result["slides"]), 0)

    def test_extract_pptx_text_only(self):
        result = extract_file(self.test_files['pptx'], prefer_text=True)
        self.assertIn("text", result)
        self.assertGreater(len(result["text"]), 0)



if __name__ == "__main__":
    unittest.main()
