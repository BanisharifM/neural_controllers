import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Set up directories
base_dir = '/work/hdd/bdau/mbanisharifdehkordi/neural_controllers'
cache_dir = os.path.join(base_dir, 'model_cache')
models_dir = os.path.join(base_dir, 'models')

# Create directories if they don't exist
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Set environment variables for HuggingFace to use our cache directory
os.environ['HF_HOME'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HUB_CACHE'] = cache_dir

print(f"Cache directory: {cache_dir}")
print(f"Models directory: {models_dir}")
print("-" * 50)

# Download Llama-3.1-8B-Instruct
print("Downloading Llama-3.1-8B-Instruct...")
model_id = "meta-llama/Llama-3.1-8B-Instruct"

try:
    # Download tokenizer
    print("Step 1/2: Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        token=os.environ.get('HF_TOKEN')
    )
    
    # Save tokenizer to models directory
    tokenizer.save_pretrained(os.path.join(models_dir, 'Llama-3.1-8B-Instruct'))
    print("✓ Tokenizer saved to models/Llama-3.1-8B-Instruct")
    
    # Download model
    print("\nStep 2/2: Downloading model weights (~16GB)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        token=os.environ.get('HF_TOKEN')
    )
    
    # Save model to models directory  
    model.save_pretrained(os.path.join(models_dir, 'Llama-3.1-8B-Instruct'))
    print("✓ Model saved to models/Llama-3.1-8B-Instruct")
    
    print("\n" + "=" * 50)
    print("SUCCESS! Llama-3.1-8B-Instruct downloaded completely.")
    print(f"Model location: {os.path.join(models_dir, 'Llama-3.1-8B-Instruct')}")
    print(f"Cache location: {cache_dir}")
    print("=" * 50)
    
except Exception as e:
    print(f"\nError occurred: {e}")
    print("Please check your HF_TOKEN and internet connection.")