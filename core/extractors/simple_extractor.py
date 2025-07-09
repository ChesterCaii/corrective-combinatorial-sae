import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


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


class SimplePromptDataset(Dataset):
    """Simple dataset for generating prompts."""
    
    def __init__(self, num_prompts: int = 1000, sequence_length: int = 256):
        """
        Initialize the prompt dataset.
        
        Args:
            num_prompts: Number of prompts to generate
            sequence_length: Length of each sequence
        """
        self.num_prompts = num_prompts
        self.sequence_length = sequence_length
        
        # Generate simple prompts
        self._generate_prompts()
    
    def _generate_prompts(self):
        """Generate simple prompts."""
        print("Generating simple prompts...")
        self.prompts = []
        
        # Create simple prompts
        base_prompts = [
            "The quick brown fox jumps over the lazy dog.",
            "In a hole in the ground there lived a hobbit.",
            "It was the best of times, it was the worst of times.",
            "To be or not to be, that is the question.",
            "All happy families are alike; each unhappy family is unhappy in its own way.",
            "Call me Ishmael.",
            "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
            "The only way to do great work is to love what you do.",
            "Life is what happens when you're busy making other plans.",
            "The future belongs to those who believe in the beauty of their dreams."
        ]
        
        for i in range(self.num_prompts):
            # Repeat and vary the base prompts
            prompt = base_prompts[i % len(base_prompts)]
            # Add some variation
            prompt += f" This is prompt number {i+1}. "
            # Repeat to reach desired length
            while len(prompt) < self.sequence_length * 2:
                prompt += prompt
            
            self.prompts.append(prompt[:self.sequence_length * 4])  # Truncate
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return self.prompts[idx]


class SimpleSAEExtractor:
    """Extract activations from a simple model."""
    
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-small",
                 target_layers: List[int] = [2, 4, 6, 8],
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the simple SAE extractor.
        
        Args:
            model_name: Name of the model
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
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        
        if device == "cpu":
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
                             num_prompts: int = 1000,
                             batch_size: int = 16,
                             sequence_length: int = 256) -> Dict[int, np.ndarray]:
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
        dataset = SimplePromptDataset(num_prompts=num_prompts, sequence_length=sequence_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Clear previous activations
        for hook in self.hooks.values():
            hook.clear()
        
        # Process batches
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Processing prompts"):
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
    """Main function to extract heldout codes."""
    # Initialize extractor with a smaller model
    extractor = SimpleSAEExtractor(
        model_name="microsoft/DialoGPT-small",
        target_layers=[2, 4, 6, 8]
    )
    
    # Extract heldout codes
    heldout_codes = extractor.extract_heldout_codes(
        num_prompts=1000,  # Smaller number for testing
        batch_size=16,
        sequence_length=256
    )
    
    # Save heldout codes
    extractor.save_heldout_codes(heldout_codes, "heldout_codes.npy")
    
    # Clean up
    extractor.remove_hooks()
    
    print("Extraction complete!")


if __name__ == "__main__":
    main() 