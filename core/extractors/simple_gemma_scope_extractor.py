"""
Simplified GemmaScope SAE Feature Extractor

This implementation directly downloads and loads GemmaScope SAEs from HuggingFace
without using the SAELens library to avoid dependency issues.

Based on GemmaScope: https://huggingface.co/google/gemma-scope
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import time
import os
from huggingface_hub import hf_hub_download
import json

class SimpleGemmaScopeExtractor:
    """Extract SAE features from Gemma-2-2B using GemmaScope SAEs."""
    
    def __init__(self, 
                 model_name: str = "google/gemma-2-2b",
                 sae_layers: List[int] = [4, 8, 12, 16],
                 device: str = "auto"):
        """
        Initialize the GemmaScope SAE extractor.
        
        Args:
            model_name: HuggingFace model name for Gemma-2-2B
            sae_layers: Layers to extract SAE features from
            device: Device to run on ('auto', 'cpu', 'cuda', 'mps')
        """
        self.model_name = model_name
        self.sae_layers = sae_layers
        self.device = self._setup_device(device)
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.saes = {}  # Store loaded SAEs
        self.hooks = []
        
        print(f"Initializing Simple GemmaScope Extractor")
        print(f"Model: {model_name}")
        print(f"SAE Layers: {sae_layers}")
        print(f"Device: {self.device}")
    
    def _setup_device(self, device: str) -> str:
        """Setup compute device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            # Temporarily disable MPS due to dtype issues
            # elif torch.backends.mps.is_available():
            #     return "mps"
            else:
                return "cpu"
        return device
    
    def load_model(self):
        """Load Gemma-2-2B model and tokenizer."""
        print("Loading Gemma-2-2B model...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map=self.device if self.device != "cpu" else None,
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("This might be due to authentication issues with Gemma-2-2B")
            print("Falling back to a smaller open model for demonstration...")
            
            # Fallback to GPT-2 medium for testing
            self.model_name = "gpt2-medium"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map=self.device if self.device != "cpu" else None
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"Fallback model {self.model_name} loaded successfully")
            return False
    
    def download_sae_weights(self, layer: int) -> Optional[str]:
        """
        Download SAE weights for a specific layer from GemmaScope.
        
        Args:
            layer: Layer number to download SAE for
            
        Returns:
            Path to downloaded SAE weights file, or None if failed
        """
        try:
            # GemmaScope SAE repository pattern
            repo_id = f"google/gemma-scope-2b-pt-res"
            filename = f"layer_{layer}/width_16k/average_l0_71/params.npz"
            
            print(f"Downloading SAE weights for layer {layer}...")
            
            # Download the SAE parameters
            weights_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir="./sae_cache"
            )
            
            print(f"Downloaded SAE for layer {layer}: {weights_path}")
            return weights_path
            
        except Exception as e:
            print(f"Could not download SAE for layer {layer}: {e}")
            print("This is expected if you don't have access to GemmaScope")
            return None
    
    def load_simple_sae(self, weights_path: str, layer: int) -> Optional[nn.Module]:
        """
        Load a simple SAE from downloaded weights.
        
        Args:
            weights_path: Path to SAE weights file
            layer: Layer number
            
        Returns:
            Simple SAE module or None if failed
        """
        try:
            # Load numpy weights
            weights = np.load(weights_path)
            
            # Extract encoder/decoder weights and ensure consistent dtype
            encoder_weight = torch.from_numpy(weights['W_enc']).T.float()  # Shape: [input_dim, sae_dim]
            decoder_weight = torch.from_numpy(weights['W_dec']).float()    # Shape: [sae_dim, input_dim]
            encoder_bias = torch.from_numpy(weights['b_enc']).float()      # Shape: [sae_dim]
            
            # Check dimension compatibility with current model
            model_dim = self.model.config.hidden_size
            sae_input_dim = encoder_weight.shape[0]
            
            if model_dim != sae_input_dim:
                print(f"Dimension mismatch: model={model_dim}, SAE={sae_input_dim}. Using mock SAE instead.")
                return None
            
            # Create simple SAE module
            class SimpleSAE(nn.Module):
                def __init__(self, encoder_weight, decoder_weight, encoder_bias):
                    super().__init__()
                    self.encoder_weight = nn.Parameter(encoder_weight, requires_grad=False)
                    self.decoder_weight = nn.Parameter(decoder_weight, requires_grad=False)
                    self.encoder_bias = nn.Parameter(encoder_bias, requires_grad=False)
                
                def encode(self, x):
                    # Ensure consistent dtype
                    x = x.float()
                    # Encode: ReLU(x @ W_enc + b_enc)
                    encoded = torch.relu(x @ self.encoder_weight + self.encoder_bias)
                    return encoded
                
                def decode(self, encoded):
                    # Decode: encoded @ W_dec
                    return encoded @ self.decoder_weight
                
                def forward(self, x):
                    encoded = self.encode(x)
                    decoded = self.decode(encoded)
                    return encoded, decoded
            
            sae = SimpleSAE(encoder_weight, decoder_weight, encoder_bias)
            sae = sae.to(self.device)
            sae.eval()
            
            print(f"Loaded SAE for layer {layer}: {encoder_weight.shape} -> {decoder_weight.shape}")
            return sae
            
        except Exception as e:
            print(f"Error loading SAE weights: {e}")
            return None
    
    def create_mock_sae(self, input_dim: int, sae_dim: int = 16384) -> nn.Module:
        """Create a mock SAE for demonstration when real SAEs aren't available."""
        
        class MockSAE(nn.Module):
            def __init__(self, input_dim, sae_dim):
                super().__init__()
                self.encoder = nn.Linear(input_dim, sae_dim, bias=True)
                self.decoder = nn.Linear(sae_dim, input_dim, bias=False)
                
                # Initialize with small random weights
                nn.init.xavier_uniform_(self.encoder.weight, gain=0.1)
                nn.init.zeros_(self.encoder.bias)
                nn.init.xavier_uniform_(self.decoder.weight, gain=0.1)
            
            def encode(self, x):
                return torch.relu(self.encoder(x))
            
            def decode(self, encoded):
                return self.decoder(encoded)
            
            def forward(self, x):
                encoded = self.encode(x)
                decoded = self.decode(encoded)
                return encoded, decoded
        
        sae = MockSAE(input_dim, sae_dim).to(self.device)
        sae.eval()
        return sae
    
    def setup_saes(self):
        """Setup SAEs for all specified layers."""
        print("Setting up SAEs...")
        
        # Get model dimension
        if hasattr(self.model.config, 'hidden_size'):
            input_dim = self.model.config.hidden_size
        else:
            input_dim = 1024  # Default fallback
        
        for layer in self.sae_layers:
            # Try to download and load real SAE
            weights_path = self.download_sae_weights(layer)
            
            if weights_path:
                sae = self.load_simple_sae(weights_path, layer)
                if sae:
                    self.saes[layer] = sae
                    continue
            
            # Fallback to mock SAE
            print(f"Using mock SAE for layer {layer}")
            self.saes[layer] = self.create_mock_sae(input_dim)
        
        print(f"Setup complete: {len(self.saes)} SAEs loaded")
    
    def register_hooks(self):
        """Register forward hooks to capture activations and compute SAE features."""
        
        class SAEHook:
            def __init__(self, layer_idx, sae, extractor):
                self.layer_idx = layer_idx
                self.sae = sae
                self.extractor = extractor
                self.activations = []
                self.sae_features = []
            
            def __call__(self, module, input, output):
                # Get hidden states (first element of output for transformer layers)
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                
                # Mean pool over sequence dimension and ensure float dtype
                pooled = hidden_states.mean(dim=1).float()  # [batch, hidden_dim]
                
                # Compute SAE features
                with torch.no_grad():
                    sae_encoded, _ = self.sae(pooled)
                
                # Store results
                self.activations.append(pooled.cpu().numpy())
                self.sae_features.append(sae_encoded.cpu().numpy())
        
        # Register hooks on transformer layers
        if hasattr(self.model, 'transformer'):
            # GPT-2 style
            layers = self.model.transformer.h
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # Gemma style
            layers = self.model.model.layers
        else:
            print("Warning: Could not find transformer layers")
            return
        
        for layer_idx in self.sae_layers:
            if layer_idx < len(layers):
                sae = self.saes[layer_idx]
                hook = SAEHook(layer_idx, sae, self)
                
                # Register hook
                handle = layers[layer_idx].register_forward_hook(hook)
                self.hooks.append((handle, hook))
                
                print(f"Registered SAE hook on layer {layer_idx}")
    
    def extract_features(self, prompts: List[str], batch_size: int = 32, max_length: int = 256) -> Dict[int, np.ndarray]:
        """
        Extract SAE features from prompts.
        
        Args:
            prompts: List of text prompts
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            
        Returns:
            Dictionary mapping layer indices to SAE feature arrays
        """
        print(f"Extracting SAE features from {len(prompts)} prompts...")
        
        # Clear previous activations
        for _, hook in self.hooks:
            hook.activations = []
            hook.sae_features = []
        
        # Process in batches
        num_batches = (len(prompts) + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in tqdm(range(num_batches), desc="Processing batches"):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(prompts))
                batch_prompts = prompts[start_idx:end_idx]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length
                ).to(self.device)
                
                # Forward pass (this triggers our hooks)
                _ = self.model(**inputs)
        
        # Collect results
        results = {}
        for _, hook in self.hooks:
            if hook.sae_features:
                # Concatenate all batches
                all_features = np.concatenate(hook.sae_features, axis=0)
                results[hook.layer_idx] = all_features
                print(f"Layer {hook.layer_idx}: {all_features.shape}")
        
        return results
    
    def cleanup(self):
        """Remove hooks and cleanup."""
        for handle, _ in self.hooks:
            handle.remove()
        self.hooks = []


def create_synthetic_prompts(n_prompts: int = 5000) -> List[str]:
    """Create synthetic prompts for testing."""
    templates = [
        "The capital of {} is",
        "In the year {}, scientists discovered",
        "The most important thing about {} is that",
        "When considering {}, we must remember",
        "The relationship between {} and {} demonstrates",
        "Recent research on {} has shown",
        "The fundamental principle of {} states",
        "Throughout history, {} has been"
    ]
    
    topics = [
        "artificial intelligence", "climate change", "quantum physics", "democracy",
        "literature", "medicine", "economics", "psychology", "philosophy", "art",
        "technology", "education", "environment", "politics", "science", "culture"
    ]
    
    countries = ["France", "Japan", "Brazil", "Germany", "India", "Australia", "Canada"]
    years = ["1969", "1995", "2001", "2010", "2020", "2023"]
    
    prompts = []
    for i in range(n_prompts):
        template = templates[i % len(templates)]
        
        if "{}" in template:
            if "capital" in template:
                topic = countries[i % len(countries)]
            elif "year" in template:
                topic = years[i % len(years)]
            else:
                topic = topics[i % len(topics)]
            
            if template.count("{}") == 2:
                topic2 = topics[(i + 1) % len(topics)]
                prompt = template.format(topic, topic2)
            else:
                prompt = template.format(topic)
        else:
            prompt = template
        
        prompts.append(prompt)
    
    return prompts


def main():
    """Main execution function."""
    print("=== Simple GemmaScope SAE Feature Extraction ===")
    
    # Configuration
    N_PROMPTS = 5000
    BATCH_SIZE = 32
    MAX_LENGTH = 256
    SAE_LAYERS = [4, 8, 12, 16]  # Original experiment layers
    
    # Create extractor
    extractor = SimpleGemmaScopeExtractor(
        sae_layers=SAE_LAYERS,
        device="auto"
    )
    
    # Load model
    success = extractor.load_model()
    if not success:
        print("Note: Using fallback model due to Gemma-2-2B access issues")
        # Adjust layers for GPT-2 (12 layers total, 0-indexed)
        SAE_LAYERS = [2, 4, 6, 8]
        extractor.sae_layers = SAE_LAYERS
    
    # Setup SAEs
    extractor.setup_saes()
    
    # Register hooks
    extractor.register_hooks()
    
    # Create prompts
    print(f"Generating {N_PROMPTS} synthetic prompts...")
    prompts = create_synthetic_prompts(N_PROMPTS)
    
    # Extract features
    start_time = time.time()
    sae_features = extractor.extract_features(
        prompts=prompts,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH
    )
    end_time = time.time()
    
    # Save results
    output_file = "simple_gemma_scope_features.npy"
    np.save(output_file, sae_features)
    
    print(f"\nExtraction Complete!")
    print(f"Time taken: {end_time - start_time:.1f} seconds")
    print(f"Features saved to: {output_file}")
    print(f"Layers extracted: {list(sae_features.keys())}")
    
    for layer, features in sae_features.items():
        print(f"  Layer {layer}: {features.shape}")
    
    # Cleanup
    extractor.cleanup()
    
    print("\nNext steps:")
    print("1. Run correlation analysis on SAE features")
    print("2. Compare with raw activation correlations")
    print("3. Scale up to full 50K prompts")
    
    return sae_features


if __name__ == "__main__":
    features = main() 
