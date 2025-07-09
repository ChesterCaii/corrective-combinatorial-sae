import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import time


class ActivationHook:
    """Hook to capture mean-pooled activations from model layers."""
    
    def __init__(self, layer_name: str):
        self.layer_name = layer_name
        self.activations = []
        
    def __call__(self, module, input, output):
        """Capture activations from the layer."""
        # For transformer layers, output is typically a tuple (hidden_states, ...)
        if isinstance(output, tuple):
            hidden_states = output[0]  # Shape: (batch_size, seq_len, hidden_size)
        else:
            hidden_states = output
            
        # Mean pool over sequence dimension
        mean_pooled = hidden_states.mean(dim=1)  # Shape: (batch_size, hidden_size)
        self.activations.append(mean_pooled.detach().cpu())
        
    def get_activations(self) -> torch.Tensor:
        """Get all captured activations."""
        if self.activations:
            return torch.cat(self.activations, dim=0)
        return torch.empty(0)
    
    def clear(self):
        """Clear stored activations."""
        self.activations.clear()


class FastPromptDataset(Dataset):
    """Fast dataset using synthetic prompts instead of loading external datasets."""
    
    def __init__(self, num_prompts: int = 5000, sequence_length: int = 256):
        """
        Initialize the fast prompt dataset.
        
        Args:
            num_prompts: Number of prompts to generate
            sequence_length: Length of each sequence
        """
        self.num_prompts = num_prompts
        self.sequence_length = sequence_length
        
        # Generate synthetic prompts quickly
        self._generate_fast_prompts()
    
    def _generate_fast_prompts(self):
        """Generate synthetic prompts quickly."""
        print(f"Generating {self.num_prompts} synthetic prompts...")
        
        # Base templates for variety
        templates = [
            "The quick brown fox jumps over the lazy dog. This is a story about",
            "In a world where technology advances rapidly, we must consider",
            "The importance of artificial intelligence in modern society cannot be",
            "Climate change represents one of the most significant challenges",
            "Education plays a crucial role in shaping the future of",
            "The development of renewable energy sources is essential for",
            "Medical breakthroughs have revolutionized the way we treat",
            "Space exploration continues to push the boundaries of human",
            "The digital revolution has transformed how we communicate and",
            "Scientific research provides the foundation for understanding"
        ]
        
        self.prompts = []
        for i in range(self.num_prompts):
            # Select template and add variation
            template = templates[i % len(templates)]
            prompt = f"{template} {i+1}. " * (self.sequence_length // 50)  # Repeat to reach desired length
            prompt = prompt[:self.sequence_length * 4]  # Truncate
            self.prompts.append(prompt)
        
        print(f"Generated {len(self.prompts)} prompts")
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return self.prompts[idx]


class FastSAEExtractor:
    """Fast SAE extractor optimized for speed."""
    
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-small",  # Smaller model for speed
                 target_layers: List[int] = [2, 4, 6, 8],  # Fewer layers
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the fast SAE extractor.
        
        Args:
            model_name: Name of the model (using smaller model for speed)
            target_layers: List of layer indices to extract activations from
            device: Device to run the model on
        """
        self.model_name = model_name
        self.target_layers = target_layers
        self.device = device
        self.hooks = {}
        
        # Load model and tokenizer
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map=None,  # Load on CPU
        )
        
        self.model = self.model.to(device)
        self.model.eval()
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks on target layers."""
        print(f"Registering hooks on layers: {self.target_layers}")
        
        # Get the transformer layers
        transformer_layers = self.model.transformer.h
        
        for layer_idx in self.target_layers:
            if layer_idx < len(transformer_layers):
                layer = transformer_layers[layer_idx]
                hook = ActivationHook(f"layer_{layer_idx}")
                
                # Register hook on the layer output
                layer.register_forward_hook(hook)
                self.hooks[layer_idx] = hook
                print(f"Registered hook on layer {layer_idx}")
            else:
                print(f"Warning: Layer {layer_idx} not found in model")
    
    def extract_heldout_codes(self, 
                             num_prompts: int = 5000,
                             batch_size: int = 64,  # Larger batch size for efficiency
                             sequence_length: int = 256) -> Dict[int, np.ndarray]:  # Shorter sequences
        """
        Extract heldout codes from the target layers.
        
        Args:
            num_prompts: Number of prompts to process
            batch_size: Batch size for processing
            sequence_length: Length of each sequence
            
        Returns:
            Dictionary mapping layer indices to heldout codes
        """
        print(f"Extracting heldout codes for {num_prompts} prompts...")
        
        # Create dataset and dataloader
        dataset = FastPromptDataset(num_prompts=num_prompts, sequence_length=sequence_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Clear previous activations
        for hook in self.hooks.values():
            hook.clear()
        
        # Process batches with timing
        start_time = time.time()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing prompts")):
                # Tokenize batch
                tokenized = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=sequence_length,
                    return_tensors="pt"
                )
                
                # Move to device
                input_ids = tokenized["input_ids"].to(self.device)
                attention_mask = tokenized["attention_mask"].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                
                # Print progress every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    batches_per_sec = (batch_idx + 1) / elapsed
                    print(f"Processed {batch_idx + 1} batches, {batches_per_sec:.2f} batches/sec")
        
        total_time = time.time() - start_time
        print(f"Total processing time: {total_time:.2f} seconds")
        
        # Collect activations and convert to heldout codes
        heldout_codes = {}
        for layer_idx, hook in self.hooks.items():
            activations = hook.get_activations()
            heldout_codes[layer_idx] = activations.numpy()
            print(f"Layer {layer_idx}: {heldout_codes[layer_idx].shape}")
        
        return heldout_codes
    
    def save_heldout_codes(self, 
                          heldout_codes: Dict[int, np.ndarray],
                          output_path: str = "heldout_codes.npy"):
        """Save heldout codes to a numpy file."""
        print(f"Saving heldout codes to {output_path}")
        np.save(output_path, heldout_codes)
        print("Heldout codes saved successfully!")
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks.values():
            hook.clear()
        self.hooks.clear()
        print("Removed all hooks")


def main():
    """Main function to extract heldout codes quickly."""
    print("=== Fast SAE Extractor ===")
    print("Optimized for speed with:")
    print("- DialoGPT-small (117M params vs 345M)")
    print("- 5K prompts (vs 50K)")
    print("- 256 sequence length (vs 512)")
    print("- Synthetic prompts (vs Wikipedia)")
    print("- Larger batch size (64 vs 32)")
    print()
    
    # Initialize extractor
    extractor = FastSAEExtractor(
        model_name="microsoft/DialoGPT-small",
        target_layers=[2, 4, 6, 8]
    )
    
    # Extract heldout codes
    heldout_codes = extractor.extract_heldout_codes(
        num_prompts=5000,
        batch_size=64,
        sequence_length=256
    )
    
    # Save heldout codes
    extractor.save_heldout_codes(heldout_codes, "fast_heldout_codes.npy")
    
    # Clean up
    extractor.remove_hooks()
    
    print("Fast extraction complete!")


if __name__ == "__main__":
    main() 