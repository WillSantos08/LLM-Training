"""
Document processor for preparing training data
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

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
    
    def _process_txt(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[Dict[str, str]]:
        """Process text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        chunks = self._chunk_text(text, chunk_size, chunk_overlap)
        
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
        """Process PDF file"""
        if PyPDF2 is None:
            print("⚠ PyPDF2 not installed. Skipping PDF processing.")
            return []
        
        text = ""
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        
        chunks = self._chunk_text(text, chunk_size, chunk_overlap)
        
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
        if docx is None:
            print("⚠ python-docx not installed. Skipping DOCX processing.")
            return []
        
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        
        chunks = self._chunk_text(text, chunk_size, chunk_overlap)
        
        return [
            {
                'question': f'What is in {file_path.name}?',
                'answer': chunk
            }
            for chunk in chunks
        ]
    
    def _process_json(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[Dict[str, str]]:
        """Process JSON file"""
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
        
        return results
    
    def _process_csv(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[Dict[str, str]]:
        """Process CSV file"""
        import csv
        
        results = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'question' in row and 'answer' in row:
                    results.append({
                        'question': row['question'],
                        'answer': row['answer']
                    })
        
        return results
    
    def _process_markdown(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[Dict[str, str]]:
        """Process Markdown file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        chunks = self._chunk_text(text, chunk_size, chunk_overlap)
        
        return [
            {
                'question': f'What is in {file_path.name}?',
                'answer': chunk
            }
            for chunk in chunks
        ]
    
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