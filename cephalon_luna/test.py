"""
Testing/Inference script for Cephalon Luna
"""

import argparse
import torch
from pathlib import Path

from config import ConfigSmall, ConfigLarge
from config.transformer import TransformerModel
from transformers import AutoTokenizer


class CephalonLunaChat:
    """Interactive chat with Cephalon Luna"""
    
    def __init__(self, model_path: str, config_size: str = 'small'):
        """
        Initialize chat interface
        
        Args:
            model_path: Path to trained model
            config_size: Configuration size ('small' or 'large')
        """
        # Load config
        if config_size == 'small':
            self.config = ConfigSmall()
        else:
            self.config = ConfigLarge()
        
        # Load tokenizer
        print(f"📦 Loading tokenizer: {self.config.tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        print(f"🧠 Loading model from: {model_path}")
        self.model = TransformerModel(self.config)
        
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.config.device)
            
            # Handle both full checkpoint and state_dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            print(f"✅ Model loaded successfully\n")
        else:
            print(f"⚠ Model file not found: {model_path}")
            print("   Using untrained model\n")
        
        self.model = self.model.to(self.config.device)
        self.model.eval()
    
    def chat(
        self,
        user_input: str,
        temperature: float = 0.8,
        max_tokens: int = 100
    ) -> str:
        """
        Generate response to user input
        
        Args:
            user_input: User's message
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        
        Returns:
            Model's response
        """
        # Format prompt
        prompt = f"User: {user_input}\nAssistant:"
        
        # Tokenize
        prompt_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        prompt_ids = prompt_ids.to(self.config.device)
        
        # Generate
        with torch.no_grad():
            generated = self.model.generate(
                prompt_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p
            )
        
        # Decode
        full_response = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "Assistant:" in full_response:
            response = full_response.split("Assistant:")[-1].strip()
        else:
            response = full_response
        
        return response
    
    def interactive_chat(self):
        """Start interactive chat session"""
        print("\n" + "="*70)
        print("CEPHALON LUNA - CHAT MODE")
        print("="*70)
        print("Type 'exit' or 'quit' to end the conversation")
        print("Type 'help' for commands")
        print("="*70 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit']:
                    print("\nGoodbye! 👋")
                    break
                
                if user_input.lower() == 'help':
                    print("\nAvailable commands:")
                    print("  exit/quit - End conversation")
                    print("  help - Show this message")
                    print()
                    continue
                
                # Generate response
                print("\nAssistant: ", end="", flush=True)
                response = self.chat(user_input)
                print(response)
                print()
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"❌ Error: {e}\n")


def main():
    """Main testing function"""
    
    parser = argparse.ArgumentParser(description='Test/Chat with Cephalon Luna')
    parser.add_argument(
        '--model',
        type=str,
        default='cephalon_luna/models/trained/final_model.pth',
        help='Path to trained model'
    )
    parser.add_argument(
        '--config',
        type=str,
        choices=['small', 'large'],
        default='small',
        help='Configuration size'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        help='Single prompt to test (without interactive mode)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Sampling temperature (0.0 to 2.0)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=100,
        help='Maximum tokens to generate'
    )
    
    args = parser.parse_args()
    
    # =========================================================================
    # INITIALIZE CHAT
    # =========================================================================
    print("\n" + "="*70)
    print("CEPHALON LUNA - TEST SCRIPT")
    print("="*70 + "\n")
    
    chat = CephalonLunaChat(args.model, args.config)
    
    # =========================================================================
    # SINGLE PROMPT OR INTERACTIVE
    # =========================================================================
    if args.prompt:
        # Single prompt mode
        print(f"You: {args.prompt}")
        response = chat.chat(
            args.prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        print(f"Assistant: {response}\n")
    else:
        # Interactive mode
        chat.interactive_chat()


if __name__ == "__main__":
    main()