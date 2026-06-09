"""
Training script for Cephalon Luna
"""

import argparse
import torch
from pathlib import Path

from config import ConfigSmall, ConfigLarge
from config.transformer import TransformerModel
from config.data import create_dataloader, Trainer


def main():
    """Main training function"""
    
    parser = argparse.ArgumentParser(description='Train Cephalon Luna')
    parser.add_argument(
        '--config',
        type=str,
        choices=['small', 'large'],
        default='small',
        help='Configuration size (small for CPU, large for GPU)'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/processed/sample_data.json',
        help='Path to training data JSON file'
    )
    parser.add_argument(
        '--raw',
        type=str,
        default=None,
        help='Path to raw text file (will be processed automatically)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (uses config default if not set)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (uses config default if not set)'
    )
    
    args = parser.parse_args()
    
    # =========================================================================
    # LOAD CONFIGURATION
    # =========================================================================
    print("\n" + "="*70)
    print("CEPHALON LUNA - TRAINING SCRIPT")
    print("="*70 + "\n")
    
    print(f"📋 Loading {args.config} configuration...")
    if args.config == 'small':
        config = ConfigSmall()
    else:
        config = ConfigLarge()
    
    print(config)
    
    # =========================================================================
    # PROCESS RAW FILE IF PROVIDED
    # =========================================================================
    if args.raw:
        print(f"📄 Processing raw file: {args.raw}")
        
        from process_text import TextProcessor
        
        processor = TextProcessor()
        filename = Path(args.raw).stem
        output_file = f"data/processed/{filename}.json"
        
        processor.process_to_training_format(args.raw, output_file)
        
        # Use processed data for training
        args.data = output_file
        print()
    
    # Override config if arguments provided
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    
    # =========================================================================
    # CHECK DATA FILE
    # =========================================================================
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ Data file not found: {args.data}")
        print(f"   Expected location: {data_path.absolute()}")
        return
    
    print(f"✅ Data file found: {args.data}\n")
    
    # =========================================================================
    # CREATE MODEL
    # =========================================================================
    print("🧠 Creating model...")
    model = TransformerModel(config)
    print()
    
    # =========================================================================
    # CREATE DATALOADER
    # =========================================================================
    print("📂 Creating dataloader...")
    dataloader, tokenizer = create_dataloader(
        file_path=args.data,
        tokenizer_name=config.tokenizer_name,
        batch_size=config.batch_size,
        max_length=config.max_sequence_length,
        shuffle=config.shuffle,
        num_workers=config.num_workers
    )
    
    if dataloader is None:
        print("❌ Failed to create dataloader")
        return
    
    print()
    
    # =========================================================================
    # CREATE TRAINER
    # =========================================================================
    print("🏋️ Creating trainer...")
    trainer = Trainer(model, config)
    print()
    
    # =========================================================================
    # START TRAINING
    # =========================================================================
    try:
        trainer.train(
            train_dataloader=dataloader,
            num_epochs=config.num_epochs
        )
        
        # Save final model
        trainer.save_model("final_model.pth")
        
        print("\n✅ Training completed successfully!")
        print(f"   Model saved to: {config.trained_models_path}/final_model.pth")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        print("Saving checkpoint...")
        trainer.save_checkpoint("interrupted.pth")
    
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()