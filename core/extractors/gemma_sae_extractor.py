import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import os
from huggingface_hub import hf_hub_download


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


class PromptDataset(Dataset):
    """Dataset for generating prompts for heldout codes."""
    
    def __init__(self, num_prompts: int = 50000, sequence_length: int = 512):
        """
        Initialize the prompt dataset.
        
        Args:
            num_prompts: Number of prompts to generate
            sequence_length: Length of each sequence
        """
        self.num_prompts = num_prompts
        self.sequence_length = sequence_length
        
        # Load a dataset for prompts (using a subset of The Pile)
        print("Loading prompt dataset...")
        self.dataset = load_dataset("EleutherAI/pile", split="train", streaming=True)
        
        # Generate prompts
        self._generate_prompts()
    
    def _generate_prompts(self):
        """Generate prompts from the dataset."""
        print("Generating prompts...")
        self.prompts = []
        
        # Take first num_prompts documents
        for i, example in enumerate(self.dataset):
            if len(self.prompts) >= self.num_prompts:
                break
                
            text = example['text']
            # Truncate to sequence_length tokens (rough approximation)
            if len(text) > self.sequence_length * 4:  # ~4 chars per token
                text = text[:self.sequence_length * 4]
            
            self.prompts.append(text)
            
            if (i + 1) % 1000 == 0:
                print(f"Generated {len(self.prompts)} prompts")
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return self.prompts[idx]


class LlamaSAEExtractor:
    """Extract activations from Llama-2-7B with SAE."""
    
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-2-7b-hf",
                 target_layers: List[int] = [4, 8, 12, 16],
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the Llama SAE extractor.
        
        Args:
            model_name: Name of the Llama model
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
            torch_dtype=torch.float16,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        
        if device == "cpu":
            self.model = self.model.to(device)
            
        self.model.eval()
        
        # Load SAE (optional - will continue without if not available)
        self._load_sae()
        
        # Register hooks
        self._register_hooks()
    
    def _load_sae(self):
        """Load SAE weights if available."""
        print("Attempting to load SAE weights...")
        
        # Try to load from a common SAE repository
        sae_repo = "neelnanda-io/llama-2-7b-sae"  # Example SAE repo
        
        try:
            # Download SAE weights
            sae_weights_path = hf_hub_download(
                repo_id=sae_repo,
                filename="sae_weights.pt",
                token=os.getenv("HF_TOKEN")
            )
            
            # Load SAE weights
            self.sae_weights = torch.load(sae_weights_path, map_location=self.device)
            print("Successfully loaded SAE weights")
            
        except Exception as e:
            print(f"Warning: Could not load SAE weights: {e}")
            print("Continuing without SAE weights - will use raw activations...")
            self.sae_weights = None
    
    def _register_hooks(self):
        """Register forward hooks on target layers."""
        print(f"Registering hooks on layers: {self.target_layers}")
        
        # Get the transformer layers
        transformer_layers = self.model.model.layers
        
        for layer_idx in self.target_layers:
            if layer_idx < len(transformer_layers):
                layer = transformer_layers[layer_idx]
                hook = ActivationHook(f"layer_{layer_idx}")
                
                # Register hook on the feed forward output
                layer.register_forward_hook(hook)
                self.hooks[layer_idx] = hook
                print(f"Registered hook on layer {layer_idx}")
            else:
                print(f"Warning: Layer {layer_idx} not found in model")
    
    def extract_heldout_codes(self, 
                             num_prompts: int = 50000,
                             batch_size: int = 32,
                             sequence_length: int = 512) -> Dict[int, np.ndarray]:
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
        dataset = PromptDataset(num_prompts=num_prompts, sequence_length=sequence_length)
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
            
            # Apply SAE if available
            if self.sae_weights is not None and f"layer_{layer_idx}" in self.sae_weights:
                # Apply SAE encoding
                sae_encoder = self.sae_weights[f"layer_{layer_idx}"]["encoder"]
                codes = torch.relu(activations @ sae_encoder.t())
            else:
                # Use raw activations as codes
                codes = activations
            
            heldout_codes[layer_idx] = codes.numpy()
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
    # Initialize extractor
    extractor = LlamaSAEExtractor(
        model_name="meta-llama/Llama-2-7b-hf",
        target_layers=[4, 8, 12, 16]
    )
    
    # Extract heldout codes
    heldout_codes = extractor.extract_heldout_codes(
        num_prompts=50000,
        batch_size=32,
        sequence_length=512
    )
    
    # Save heldout codes
    extractor.save_heldout_codes(heldout_codes, "heldout_codes.npy")
    
    # Clean up
    extractor.remove_hooks()
    
    print("Extraction complete!")


if __name__ == "__main__":
    main() 