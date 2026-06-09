"""
Document processor for preparing training data
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import docx
except ImportError:
    docx = None


class DocumentProcessor:
    """
    Process various document formats and extract text for training
    """
    
    def __init__(self):
        """Initialize document processor"""
        self.supported_formats = {
            '.pdf': self._process_pdf,
            '.txt': self._process_txt,
            '.docx': self._process_docx,
            '.json': self._process_json,
            '.csv': self._process_csv,
            '.md': self._process_markdown,
        }
    
    def process_documents(
        self,
        input_dir: str,
        output_file: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        """
        Process all documents in directory and save as JSON
        
        Args:
            input_dir: Directory containing documents
            output_file: Output JSON file path
            chunk_size: Size of text chunks (in words)
            chunk_overlap: Overlap between chunks
        """
        input_path = Path(input_dir)
        
        if not input_path.exists():
            print(f"⚠ Input directory not found: {input_dir}")
            return
        
        all_documents = []
        
        # Process each file
        for file_path in input_path.rglob('*'):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                
                if ext in self.supported_formats:
                    print(f"📄 Processing: {file_path.name}")
                    processor = self.supported_formats[ext]
                    
                    try:
                        docs = processor(file_path, chunk_size, chunk_overlap)
                        all_documents.extend(docs)
                    except Exception as e:
                        print(f"❌ Error processing {file_path.name}: {e}")
        
        # Save to JSON
        if all_documents:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_documents, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Saved {len(all_documents)} documents to {output_file}")
        else:
            print("⚠ No documents found to process")
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text
        
        Args:
            text: Raw text
        
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove extra newlines but keep some structure
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove null characters and escape sequences
        text = text.replace('\x00', '')
        text = text.replace('\\x', '')
        
        # Remove special PDF characters
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        return text.strip()
    
    def _is_valid_text(self, text: str) -> bool:
        """
        Check if extracted text is valid (not corrupted/encoded)
        
        Args:
            text: Text to validate
        
        Returns:
            True if text is valid, False if corrupted
        """
        if not text or not text.strip():
            return False
        
        # Count valid characters (letters, numbers, spaces, common punctuation)
        valid_chars = 0
        
        for c in text:
            # Check if character is valid
            if (c.isalnum() or 
                c.isspace() or 
                c in '.,!?;:\'"()-–—…«»""\'\'' or
                ord(c) > 127):  # Allow accented characters
                valid_chars += 1
        
        total_chars = len(text)
        
        if total_chars == 0:
            return False
        
        valid_ratio = valid_chars / total_chars
        
        # If less than 50% valid characters, probably corrupted
        if valid_ratio < 0.5:
            print(f"      (Text quality: {valid_ratio*100:.1f}% - likely corrupted)")
            return False
        
        return True
    
    def _process_txt(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[Dict[str, str]]:
        """Process text file"""
        print(f"   📝 Using TXT processor")
        
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        text = None
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    text = f.read()
                print(f"   ✅ Loaded with encoding: {encoding}")
                break
            except UnicodeDecodeError:
                continue
        
        if text is None:
            print(f"   ❌ Could not decode file")
            return []
        
        # Clean and chunk
        text = self._clean_text(text)
        chunks = self._chunk_text(text, chunk_size, chunk_overlap)
        
        if not chunks:
            return []
        
        print(f"   ✅ Created {len(chunks)} chunks")
        
        return [
            {
                'question': f'What is in {file_path.name}?',
                'answer': chunk
            }
            for chunk in chunks
        ]
    
    def _process_pdf(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[Dict[str, str]]:
        """
        Process PDF file with pdfplumber (preferred) or PyPDF2 (fallback)
        Handles digital text PDFs only
        """
        text = ""
        num_pages = 0
        extraction_method = None
        
        # Try pdfplumber first (better quality for digital text)
        if pdfplumber is not None:
            print(f"   📄 Using pdfplumber")
            try:
                with pdfplumber.open(file_path) as pdf:
                    num_pages = len(pdf.pages)
                    print(f"   📄 PDF has {num_pages} pages")
                    
                    for page_num, page in enumerate(pdf.pages):
                        try:
                            # Extract text
                            page_text = page.extract_text()
                            
                            if page_text and page_text.strip():
                                # Validate text quality
                                if self._is_valid_text(page_text):
                                    text += page_text + "\n"
                                    extraction_method = "pdfplumber"
                                else:
                                    print(f"   ⚠ Page {page_num + 1}: Text appears corrupted or encoded")
                            else:
                                print(f"   ⚠ Page {page_num + 1}: No text extracted")
                        
                        except Exception as e:
                            print(f"   ⚠ Error on page {page_num + 1}: {e}")
                            continue
                
                if text and text.strip():
                    print(f"   ✅ pdfplumber extracted {len(text)} characters")
                else:
                    print(f"   ⚠ pdfplumber extracted no valid text, trying PyPDF2...")
                    text = ""  # Reset for PyPDF2
            
            except Exception as e:
                print(f"   ⚠ pdfplumber error: {e}, trying PyPDF2...")
                text = ""
        
        # Fallback to PyPDF2
        if not text and PyPDF2 is not None:
            print(f"   📄 Using PyPDF2")
            try:
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    
                    if not num_pages:
                        num_pages = len(pdf_reader.pages)
                        print(f"   📄 PDF has {num_pages} pages")
                    
                    for page_num, page in enumerate(pdf_reader.pages):
                        try:
                            page_text = page.extract_text()
                            
                            if page_text and page_text.strip():
                                # Validate text quality
                                if self._is_valid_text(page_text):
                                    text += page_text + "\n"
                                    extraction_method = "PyPDF2"
                                else:
                                    print(f"   ⚠ Page {page_num + 1}: Text appears corrupted")
                            else:
                                print(f"   ⚠ Page {page_num + 1}: No text extracted")
                        
                        except Exception as e:
                            print(f"   ⚠ Error on page {page_num + 1}: {e}")
                            continue
                    
                    if text and text.strip():
                        print(f"   ✅ PyPDF2 extracted {len(text)} characters")
            
            except Exception as e:
                print(f"   ❌ PyPDF2 error: {e}")
                return []
        
        # If still no text
        if not text or not text.strip():
            print(f"   ❌ No valid text could be extracted from PDF")
            print(f"   💡 PDF might be:")
            print(f"      - Scanned (image-based) - requires OCR")
            print(f"      - Encrypted - needs decryption")
            print(f"      - Corrupted - try another PDF")
            return []
        
        # Clean and chunk
        text = self._clean_text(text)
        chunks = self._chunk_text(text, chunk_size, chunk_overlap)
        
        if not chunks:
            print(f"   ❌ No valid chunks created from PDF")
            return []
        
        print(f"   ✅ Created {len(chunks)} chunks ({extraction_method})")
        
        return [
            {
                'question': f'What is in {file_path.name}?',
                'answer': chunk
            }
            for chunk in chunks
        ]
    
    def _process_docx(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[Dict[str, str]]:
        """Process DOCX file"""
        print(f"   📝 Using DOCX processor")
        
        if docx is None:
            print("⚠ python-docx not installed. Skipping DOCX processing.")
            return []
        
        try:
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            
            if not text or not text.strip():
                print(f"   ⚠ No text found in DOCX")
                return []
            
            # Clean and chunk
            text = self._clean_text(text)
            chunks = self._chunk_text(text, chunk_size, chunk_overlap)
            
            if not chunks:
                return []
            
            print(f"   ✅ Created {len(chunks)} chunks")
            
            return [
                {
                    'question': f'What is in {file_path.name}?',
                    'answer': chunk
                }
                for chunk in chunks
            ]
        
        except Exception as e:
            print(f"   ❌ Error processing DOCX: {e}")
            return []
    
    def _process_json(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[Dict[str, str]]:
        """Process JSON file"""
        print(f"   📝 Using JSON processor")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            results = []
            
            # Handle list of objects
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        if 'question' in item and 'answer' in item:
                            results.append(item)
                        elif 'text' in item:
                            chunks = self._chunk_text(item['text'], chunk_size, chunk_overlap)
                            for chunk in chunks:
                                results.append({
                                    'question': f'From {file_path.name}:',
                                    'answer': chunk
                                })
            
            if not results:
                print(f"   ⚠ No valid data found in JSON")
                return []
            
            print(f"   ✅ Extracted {len(results)} items")
            return results
        
        except Exception as e:
            print(f"   ❌ Error processing JSON: {e}")
            return []
    
    def _process_csv(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[Dict[str, str]]:
        """Process CSV file"""
        import csv
        
        print(f"   📝 Using CSV processor")
        
        results = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'question' in row and 'answer' in row:
                        results.append({
                            'question': row['question'],
                            'answer': row['answer']
                        })
            
            if not results:
                print(f"   ⚠ No valid rows found in CSV")
                return []
            
            print(f"   ✅ Extracted {len(results)} rows")
            return results
        
        except Exception as e:
            print(f"   ❌ Error processing CSV: {e}")
            return []
    
    def _process_markdown(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[Dict[str, str]]:
        """Process Markdown file"""
        print(f"   📝 Using Markdown processor")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            if not text or not text.strip():
                print(f"   ⚠ No text found in Markdown")
                return []
            
            # Clean and chunk
            text = self._clean_text(text)
            chunks = self._chunk_text(text, chunk_size, chunk_overlap)
            
            if not chunks:
                return []
            
            print(f"   ✅ Created {len(chunks)} chunks")
            
            return [
                {
                    'question': f'What is in {file_path.name}?',
                    'answer': chunk
                }
                for chunk in chunks
            ]
        
        except Exception as e:
            print(f"   ❌ Error processing Markdown: {e}")
            return []
    
    def _chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        Divide text into overlapping chunks
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk (in words)
            chunk_overlap: Overlap between chunks (in words)
        
        Returns:
            List of text chunks
        """
        # Clean text
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Split into words
        words = text.split()
        
        if len(words) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            
            if end >= len(words):
                break
            
            start = end - chunk_overlap
        
        return chunks