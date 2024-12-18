# document_processor.py
import os
import PyPDF2
from config import PDF_DIRECTORY

def extract_text_from_pdfs(pdf_directory):
    text_chunks = []
    for filename in os.listdir(pdf_directory):
        if filename.endswith('.pdf'):
            with open(os.path.join(pdf_directory, filename), 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_number, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        text_chunks.append({
                            'text': text,
                            'metadata': {
                                'filename': filename,
                                'page_number': page_number + 1
                            }
                        })
    return text_chunks