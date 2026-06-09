"""
Process any text file into training JSON format
"""

import json
import re
from pathlib import Path
from typing import List


class TextProcessor:
    """Process raw text files into training data"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Args:
            chunk_size: Size of text chunks in words
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def read_file(self, file_path: str) -> str:
        """
        Read text from file
        
        Args:
            file_path: Path to text file
        
        Returns:
            File content as string
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        print(f"📖 Reading file: {file_path.name}")
        
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    text = f.read()
                
                print(f"✅ File loaded: {len(text)} characters (encoding: {encoding})")
                return text
            
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"Could not decode file: {file_path}")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text
        
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove extra newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove special characters (keep basic punctuation)
        text = re.sub(r'[^\w\s.,!?;:()\[\]{}\"\'\-]', ' ', text)
        
        # Normalize spaces
        text = ' '.join(text.split())
        
        return text.strip()
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to split
        
        Returns:
            List of text chunks
        """
        words = text.split()
        
        if len(words) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            
            if end >= len(words):
                break
            
            start = end - self.chunk_overlap
        
        return chunks
    
    def process_to_training_format(
        self,
        input_file: str,
        output_file: str = None
    ) -> List[dict]:
        """
        Convert text file to training JSON format
        
        Args:
            input_file: Path to input text file
            output_file: Path to output JSON file (optional)
        
        Returns:
            List of training examples
        """
        # Read text
        text = self.read_file(input_file)
        
        # Clean text
        text = self.clean_text(text)
        print(f"✅ Text cleaned: {len(text)} characters")
        
        # Chunk text
        chunks = self.chunk_text(text)
        print(f"✅ Divided into {len(chunks)} chunks")
        
        # Create training examples
        filename = Path(input_file).stem  # File name without extension
        training_data = []
        
        for i, chunk in enumerate(chunks):
            # Create different question types for variety
            if i % 3 == 0:
                question = f"What is in {filename}?"
            elif i % 3 == 1:
                question = f"Explain the content of {filename}"
            else:
                question = f"Tell me about {filename}"
            
            training_data.append({
                "question": question,
                "answer": chunk
            })
        
        # Save to JSON
        if output_file is None:
            output_file = f"data/processed/{filename}.json"
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Training data saved: {output_path}")
        print(f"✅ Total examples: {len(training_data)}")
        
        return training_data


def main():
    """Process text file for training"""
    import sys
    
    # Get input file
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = input("📂 Enter file path to process: ").strip()
    
    if not input_file:
        print("❌ No file provided")
        return
    
    # Get output file (optional)
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        filename = Path(input_file).stem
        output_file = f"data/processed/{filename}.json"
    
    # Process
    processor = TextProcessor(chunk_size=512, chunk_overlap=50)
    
    try:
        training_data = processor.process_to_training_format(
            input_file=input_file,
            output_file=output_file
        )
        
        # Show preview
        print("\n" + "="*70)
        print("PREVIEW OF TRAINING DATA")
        print("="*70)
        for i, item in enumerate(training_data[:3]):
            print(f"\nExample {i+1}:")
            print(f"Q: {item['question']}")
            print(f"A: {item['answer'][:150]}...")
        
        print("\n" + "="*70)
        print("✅ READY TO TRAIN!")
        print("="*70)
        print(f"\nRun: python train.py --data {output_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()