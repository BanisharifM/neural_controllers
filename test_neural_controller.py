import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set environment variables
os.environ['HF_TOKEN'] = os.environ.get('HF_TOKEN', '')
os.environ['HF_HOME'] = '/work/hdd/bdau/mbanisharifdehkordi/neural_controllers/model_cache'

# Add the neural_controllers directory to Python path
sys.path.insert(0, '/work/hdd/bdau/mbanisharifdehkordi/neural_controllers')

# Import the NeuralController
try:
    from neural_controllers import NeuralController
    print("✓ NeuralController imported successfully")
except ImportError as e:
    print(f"✗ Failed to import NeuralController: {e}")
    sys.exit(1)

# Test loading the model
print("\nTesting model loading...")
model_path = "/work/hdd/bdau/mbanisharifdehkordi/neural_controllers/models/Llama-3.1-8B-Instruct"

try:
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("✓ Tokenizer loaded")
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✓ Using device: {device}")
    
    if device == "cpu":
        print("⚠ Warning: Running on CPU. For better performance, allocate a GPU node.")
    
    # Create a simple test
    print("\n✓ All imports and basic tests passed!")
    print("\nNext steps:")
    print("1. If on CPU, allocate a GPU node for actual experiments")
    print("2. Run one of the example notebooks in the notebooks/ directory")
    
except Exception as e:
    print(f"✗ Error: {e}") 
