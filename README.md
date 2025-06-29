# Gemma-2-2B SAE Heldout Codes Extractor

This repository contains code to load the Gemma-2-2B model with its pretrained SAE from GemmaScope, register forward hooks on specific transformer layers, and extract heldout codes for 50K prompts.

## Features

- Loads Gemma-2-2B model from HuggingFace
- Attempts to load pretrained SAE weights from GemmaScope
- Registers forward hooks on layers 4, 8, 12, and 16
- Mean-pools activations over sequence dimension
- Processes 50K prompts through the model
- Outputs heldout codes as `heldout_codes.npy`

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up HuggingFace token (if needed for accessing GemmaScope):
```bash
export HF_TOKEN="your_huggingface_token_here"
```

## Usage

### Basic Usage

Run the main extraction script:
```bash
python gemma_sae_extractor.py
```

This will:
1. Load the Gemma-2-2B model
2. Attempt to load GemmaScope SAE weights
3. Register hooks on layers 4, 8, 12, and 16
4. Process 50K prompts from The Pile dataset
5. Save heldout codes to `heldout_codes.npy`

### Custom Usage

You can also use the classes directly:

```python
from gemma_sae_extractor import GemmaSAEExtractor

# Initialize extractor
extractor = GemmaSAEExtractor(
    model_name="google/gemma-2-2b",
    target_layers=[4, 8, 12, 16]
)

# Extract heldout codes
heldout_codes = extractor.extract_heldout_codes(
    num_prompts=50000,
    batch_size=32,
    sequence_length=512
)

# Save to file
extractor.save_heldout_codes(heldout_codes, "heldout_codes.npy")
```

## Output Format

The `heldout_codes.npy` file contains a dictionary with the following structure:
```python
{
    4: np.ndarray,  # Shape: (num_prompts, hidden_size)
    8: np.ndarray,  # Shape: (num_prompts, hidden_size)
    12: np.ndarray, # Shape: (num_prompts, hidden_size)
    16: np.ndarray  # Shape: (num_prompts, hidden_size)
}
```

## Configuration

You can modify the following parameters in the code:

- `model_name`: The HuggingFace model name
- `target_layers`: List of layer indices to extract from
- `num_prompts`: Number of prompts to process
- `batch_size`: Batch size for processing
- `sequence_length`: Maximum sequence length
- `sae_repo`: GemmaScope repository name (may need adjustment)

## Notes

- The code attempts to load GemmaScope SAE weights but will continue without them if unavailable
- Make sure you have sufficient GPU memory for the model
- The script uses streaming datasets to avoid loading the entire Pile dataset into memory
- You may need to adjust the GemmaScope repository name based on the actual release

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.35+
- CUDA-compatible GPU (recommended)
- HuggingFace account (for accessing GemmaScope) 