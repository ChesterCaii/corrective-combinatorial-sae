"""
Real GemmaScope SAE Feature Extractor using SAELens

This implementation uses the actual SAELens library to load pretrained
GemmaScope sparse autoencoders and extract features from Gemma-2-2B.

Based on:
- GemmaScope paper: https://arxiv.org/abs/2408.05147
- SAELens library: https://github.com/jbloomAus/SAELens
- GemmaScope HuggingFace: https://huggingface.co/google/gemma-scope
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
import time
import os
import json
from sae_lens import SAE  # SAELens library for loading GemmaScope SAEs
from huggingface_hub import hf_hub_download


class GemmaScopeActivationHook:
    """Hook to capture activations and apply GemmaScope SAE encoding."""
    
    def __init__(self, layer_name: str, layer_idx: int, sae: Optional[SAE] = None):
        self.layer_name = layer_name
        self.layer_idx = layer_idx
        self.sae = sae
        self.activations = []
        self.sae_features = []
        
    def __call__(self, module, input, output):
        """Capture activations and compute SAE features."""
        # For Gemma transformer layers, output format depends on layer type
        if isinstance(output, tuple):
            hidden_states = output[0]  # Shape: (batch_size, seq_len, hidden_size)
        else:
            hidden_states = output
            
        # Mean pool over sequence dimension
        mean_pooled = hidden_states.mean(dim=1)  # Shape: (batch_size, hidden_size)
        self.activations.append(mean_pooled.detach().cpu())
        
        # Apply SAE if available
        if self.sae is not None:
            with torch.no_grad():
                # Move to SAE device and apply
                device = next(self.sae.parameters()).device
                sae_input = mean_pooled.to(device)
                
                # Get SAE feature activations
                sae_output = self.sae.encode(sae_input)
                self.sae_features.append(sae_output.detach().cpu())
        
    def get_activations(self) -> torch.Tensor:
        """Get all captured raw activations."""
        if self.activations:
            return torch.cat(self.activations, dim=0)
        return torch.empty(0)
    
    def get_sae_features(self) -> torch.Tensor:
        """Get all captured SAE features."""
        if self.sae_features:
            return torch.cat(self.sae_features, dim=0)
        return torch.empty(0)
    
    def clear(self):
        """Clear stored activations."""
        self.activations.clear()
        self.sae_features.clear()


class WikiTextDataset(Dataset):
    """Dataset for sampling from WikiText for the experiment."""
    
    def __init__(self, num_tokens: int = 1_000_000, sequence_length: int = 512):
        """
        Initialize WikiText dataset for target token count.
        
        Args:
            num_tokens: Total number of tokens to sample
            sequence_length: Length of each sequence
        """
        self.num_tokens = num_tokens
        self.sequence_length = sequence_length
        self.num_sequences = num_tokens // sequence_length
        
        print(f"Loading WikiText dataset for {num_tokens:,} tokens ({self.num_sequences:,} sequences)...")
        
        # Load WikiText-103 dataset
        try:
            self.dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=False)
        except Exception as e:
            print(f"Warning: Could not load wikitext dataset: {e}")
            print("Falling back to OpenWebText...")
            try:
                self.dataset = load_dataset("openwebtext", split="train", streaming=True)
            except Exception as e2:
                print(f"Warning: Could not load openwebtext: {e2}")
                print("Using synthetic data...")
                self._create_synthetic_data()
                return
        
        # Generate sequences
        self._generate_sequences()
    
    def _create_synthetic_data(self):
        """Create synthetic text data as fallback."""
        print("Creating synthetic text data...")
        
        # Create diverse synthetic prompts
        templates = [
            "The history of {} is fascinating because {}.",
            "In the field of {}, researchers have discovered that {}.",
            "When considering {}, it's important to note that {}.",
            "The relationship between {} and {} demonstrates {}.",
            "Recent studies in {} suggest that {}.",
            "Throughout history, {} has played a crucial role in {}.",
            "The development of {} revolutionized our understanding of {}.",
            "Scientists working on {} have found evidence that {}.",
        ]
        
        topics = [
            "artificial intelligence", "quantum physics", "molecular biology", "astronomy",
            "climate science", "neuroscience", "computer science", "mathematics",
            "psychology", "economics", "literature", "philosophy", "history",
            "chemistry", "engineering", "medicine", "geography", "sociology"
        ]
        
        explanations = [
            "it involves complex mathematical principles",
            "it connects multiple disciplines",
            "it has practical applications in daily life",
            "it challenges our existing understanding",
            "it opens new research possibilities",
            "it demonstrates fundamental natural laws",
            "it has evolved significantly over time",
            "it requires interdisciplinary collaboration"
        ]
        
        self.sequences = []
        for _ in range(self.num_sequences):
            template = np.random.choice(templates)
            if "{}" in template and template.count("{}") == 2:
                topic = np.random.choice(topics)
                explanation = np.random.choice(explanations)
                text = template.format(topic, explanation)
            else:
                topic = np.random.choice(topics)
                text = template.format(topic)
            
            # Pad to approximate sequence length
            while len(text) < self.sequence_length * 3:  # ~3 chars per token
                text += " " + np.random.choice(explanations)
            
            self.sequences.append(text[:self.sequence_length * 4])  # ~4 chars per token max
        
        print(f"Generated {len(self.sequences)} synthetic sequences")
    
    def _generate_sequences(self):
        """Generate sequences from the loaded dataset."""
        print(f"Generating {self.num_sequences:,} sequences from WikiText...")
        self.sequences = []
        current_text = ""
        
        # Handle both streaming and non-streaming datasets
        if hasattr(self.dataset, '__iter__'):
            dataset_iter = iter(self.dataset)
        else:
            dataset_iter = self.dataset
        
        for i, example in enumerate(dataset_iter):
            if 'text' in example:
                text = example['text']
            elif 'content' in example:
                text = example['content']
            else:
                continue
                
            if len(text.strip()) < 10:  # Skip very short texts
                continue
                
            current_text += " " + text.strip()
            
            # Split into sequences when we have enough text
            while len(current_text) > self.sequence_length * 4:  # ~4 chars per token
                sequence = current_text[:self.sequence_length * 4]
                self.sequences.append(sequence)
                current_text = current_text[self.sequence_length * 2:]  # Overlap
                
                if len(self.sequences) >= self.num_sequences:
                    break
            
            if len(self.sequences) >= self.num_sequences:
                break
                
            if (i + 1) % 1000 == 0:
                print(f"Processed {i+1} documents, generated {len(self.sequences)} sequences")
        
        print(f"Final dataset: {len(self.sequences)} sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]


class RealGemmaScopeExtractor:
    """Extract features using real GemmaScope SAEs via SAELens."""
    
    def __init__(self, 
                 model_name: str = "google/gemma-2-2b",
                 target_layers: List[int] = [4, 8, 12, 16],
                 sae_width: int = 16384,  # 2^14
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the real GemmaScope SAE extractor.
        
        Args:
            model_name: Gemma model name
            target_layers: List of layer indices for SAE extraction
            sae_width: SAE width (16k by default for all layers)
            device: Device to run on
        """
        self.model_name = model_name
        self.target_layers = target_layers
        self.sae_width = sae_width
        self.device = device
        self.hooks = {}
        self.saes = {}
        
        print("🚀 Initializing Real GemmaScope SAE Extractor...")
        print(f"Model: {model_name}")
        print(f"Target layers: {target_layers}")
        print(f"SAE width: {sae_width}")
        print(f"Device: {device}")
        
        # Load model and tokenizer
        self._load_model()
        
        # Load GemmaScope SAEs
        self._load_gemmascope_saes()
        
        # Register hooks
        self._register_hooks()
    
    def _load_model(self):
        """Load the Gemma-2-2B model and tokenizer."""
        print("Loading Gemma-2-2B model...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=os.getenv("HF_TOKEN")
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                token=os.getenv("HF_TOKEN")
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
                
            self.model.eval()
            print("✅ Gemma-2-2B model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading Gemma-2-2B: {e}")
            print("This might be due to:")
            print("1. Missing HuggingFace token (set HF_TOKEN environment variable)")
            print("2. No access to Gemma-2-2B (request access on HuggingFace)")
            print("3. Insufficient disk space or memory")
            raise
    
    def _load_gemmascope_saes(self):
        """Load pretrained GemmaScope SAEs using SAELens."""
        print("Loading GemmaScope SAEs using SAELens...")
        
        # GemmaScope repository for 2B residual SAEs
        sae_repo = "google/gemma-scope-2b-pt-res"
        
        for layer_idx in self.target_layers:
            try:
                print(f"Loading SAE for layer {layer_idx}...")
                
                # SAE ID format for GemmaScope
                sae_id = f"{sae_repo}/layer_{layer_idx}/width_{self.sae_width//1000}k/average_l0_71"
                
                # Load SAE using SAELens
                try:
                    sae = SAE.from_pretrained(
                        release=sae_repo,
                        sae_id=f"layer_{layer_idx}",
                        device=self.device
                    )
                    self.saes[layer_idx] = sae
                    print(f"✅ Layer {layer_idx} SAE loaded: input_dim={sae.d_in}, output_dim={sae.d_sae}")
                    
                except Exception as sae_error:
                    print(f"⚠️  Could not load SAE for layer {layer_idx} via SAELens: {sae_error}")
                    print(f"Trying alternative loading method...")
                    
                    # Alternative: direct HuggingFace download
                    try:
                        sae_path = hf_hub_download(
                            repo_id=sae_repo,
                            filename=f"layer_{layer_idx}/width_{self.sae_width//1000}k/sae_weights.safetensors",
                            token=os.getenv("HF_TOKEN")
                        )
                        # This would require manual SAE reconstruction
                        print(f"Downloaded SAE weights to {sae_path}")
                        print("Manual SAE loading not implemented - skipping this layer")
                        
                    except Exception as download_error:
                        print(f"⚠️  Could not download SAE for layer {layer_idx}: {download_error}")
                        print("Continuing without SAE for this layer...")
                
            except Exception as e:
                print(f"⚠️  Error loading SAE for layer {layer_idx}: {e}")
                print("Continuing without SAE for this layer...")
        
        if self.saes:
            print(f"✅ Successfully loaded {len(self.saes)} GemmaScope SAEs")
        else:
            print("⚠️  No SAEs loaded - will extract raw activations only")
    
    def _register_hooks(self):
        """Register forward hooks on target layers."""
        print(f"Registering hooks on layers: {self.target_layers}")
        
        # Gemma-2-2B has 26 layers (layers 0-25)
        transformer_layers = self.model.model.layers
        
        for layer_idx in self.target_layers:
            if layer_idx < len(transformer_layers):
                layer = transformer_layers[layer_idx]
                sae = self.saes.get(layer_idx, None)
                hook = GemmaScopeActivationHook(f"layer_{layer_idx}", layer_idx, sae)
                
                # Register hook on the layer output
                layer.register_forward_hook(hook)
                self.hooks[layer_idx] = hook
                print(f"✅ Registered hook on layer {layer_idx} (SAE: {'Yes' if sae else 'No'})")
            else:
                print(f"⚠️  Warning: Layer {layer_idx} not found (model has {len(transformer_layers)} layers)")
    
    def extract_features(self, 
                        num_tokens: int = 1_000_000,
                        batch_size: int = 16,  # Conservative for Gemma-2-2B
                        sequence_length: int = 512) -> Dict[str, Dict[int, np.ndarray]]:
        """
        Extract features for the full experiment.
        
        Args:
            num_tokens: Total number of tokens to process
            batch_size: Batch size for processing
            sequence_length: Length of each sequence
            
        Returns:
            Dictionary with 'activations' and 'sae_features' for each layer
        """
        print(f"🎯 Starting GemmaScope feature extraction for {num_tokens:,} tokens...")
        
        # Create dataset and dataloader
        dataset = WikiTextDataset(num_tokens=num_tokens, sequence_length=sequence_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Clear previous data
        for hook in self.hooks.values():
            hook.clear()
        
        # Process batches with timing
        start_time = time.time()
        total_batches = len(dataloader)
        processed_tokens = 0
        
        print(f"Processing {total_batches} batches...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting features")):
                # Tokenize batch
                try:
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
                    
                    # Forward pass through Gemma-2-2B
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        use_cache=False  # Save memory
                    )
                    
                    # Count processed tokens
                    processed_tokens += input_ids.numel()
                    
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {e}")
                    continue
                
                # Progress reporting
                if (batch_idx + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    batches_per_sec = (batch_idx + 1) / elapsed
                    eta_minutes = (total_batches - batch_idx - 1) / batches_per_sec / 60
                    
                    print(f"Progress: {batch_idx+1}/{total_batches} batches "
                          f"({processed_tokens:,} tokens), ETA: {eta_minutes:.1f} min")
        
        total_time = time.time() - start_time
        print(f"✅ Completed in {total_time/60:.1f} minutes")
        print(f"📊 Processed {processed_tokens:,} tokens total")
        
        # Collect results
        results = {
            'activations': {},
            'sae_features': {}
        }
        
        for layer_idx, hook in self.hooks.items():
            activations = hook.get_activations()
            sae_features = hook.get_sae_features()
            
            if activations.numel() > 0:
                results['activations'][layer_idx] = activations.numpy()
                print(f"Layer {layer_idx} activations: {results['activations'][layer_idx].shape}")
            
            if sae_features.numel() > 0:
                results['sae_features'][layer_idx] = sae_features.numpy()
                print(f"Layer {layer_idx} SAE features: {results['sae_features'][layer_idx].shape}")
            else:
                print(f"Layer {layer_idx}: No SAE features (SAE not loaded)")
        
        return results
    
    def save_features(self, 
                     features: Dict[str, Dict[int, np.ndarray]],
                     output_dir: str = "gemmascope_experiment_outputs"):
        """Save extracted features."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save activations
        if features['activations']:
            activations_path = os.path.join(output_dir, "activations.npy")
            np.save(activations_path, features['activations'])
            print(f"💾 Saved activations to {activations_path}")
        
        # Save SAE features if available
        if features['sae_features']:
            sae_path = os.path.join(output_dir, "sae_features.npy")
            np.save(sae_path, features['sae_features'])
            print(f"💾 Saved SAE features to {sae_path}")
        
        # Save metadata
        metadata = {
            'model': self.model_name,
            'layers': self.target_layers,
            'sae_width': self.sae_width,
            'total_sequences': sum(len(arr) for arr in features['activations'].values()) // len(self.target_layers) if features['activations'] else 0,
            'activation_dims': {layer: arr.shape[1] for layer, arr in features['activations'].items()},
            'sae_feature_dims': {layer: arr.shape[1] for layer, arr in features['sae_features'].items()},
            'has_sae_features': bool(features['sae_features']),
            'extraction_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"📄 Saved metadata to {metadata_path}")
        
        # Save summary
        summary = [
            "🧪 GEMMASCOPE FEATURE EXTRACTION SUMMARY",
            "=" * 50,
            f"Model: {self.model_name}",
            f"Layers: {self.target_layers}",
            f"SAE Width: {self.sae_width}",
            "",
            "Results:",
        ]
        
        for layer_idx in self.target_layers:
            summary.append(f"  Layer {layer_idx}:")
            if layer_idx in features['activations']:
                summary.append(f"    Activations: {features['activations'][layer_idx].shape}")
            if layer_idx in features['sae_features']:
                summary.append(f"    SAE Features: {features['sae_features'][layer_idx].shape}")
        
        summary_text = "\n".join(summary)
        summary_path = os.path.join(output_dir, "extraction_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(summary_text)
        
        print(summary_text)
        print(f"📄 Summary saved to {summary_path}")
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks.values():
            hook.clear()
        self.hooks.clear()
        print("🧹 Removed all hooks")


def main():
    """Main function for real GemmaScope feature extraction."""
    print("=" * 60)
    print("🧪 REAL GEMMASCOPE FEATURE EXTRACTION")
    print("=" * 60)
    print("📋 Using:")
    print("- Gemma-2-2B model")
    print("- Real GemmaScope SAEs via SAELens")
    print("- Target layers: 4, 8, 12, 16")
    print("- 1M token experiment scale")
    print()
    
    # Check if HuggingFace token is set
    if not os.getenv("HF_TOKEN"):
        print("⚠️  Warning: HF_TOKEN environment variable not set")
        print("You may encounter authentication errors with Gemma-2-2B")
        print("Set your token with: export HF_TOKEN=your_token_here")
        print()
    
    # Initialize extractor
    try:
        extractor = RealGemmaScopeExtractor(
            model_name="google/gemma-2-2b",
            target_layers=[4, 8, 12, 16],
            sae_width=16384  # 16k features
        )
    except Exception as e:
        print(f"❌ Failed to initialize extractor: {e}")
        return
    
    # Extract features
    try:
        # Start with smaller scale for testing
        test_tokens = 50_000  # 50K tokens for initial test
        print(f"🧪 Starting with {test_tokens:,} tokens for testing...")
        
        features = extractor.extract_features(
            num_tokens=test_tokens,
            batch_size=8,  # Conservative batch size
            sequence_length=512
        )
        
        # Save results
        extractor.save_features(features)
        
        print("🎉 Feature extraction completed successfully!")
        print("🔍 Ready for correlation analysis with correlation_analysis.py")
        
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        
    finally:
        # Clean up
        extractor.remove_hooks()


if __name__ == "__main__":
    main() 