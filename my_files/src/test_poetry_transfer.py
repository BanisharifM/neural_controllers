import os
import sys
import torch
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM

# Initialize wandb
wandb.init(
    project=os.environ.get("WANDB_PROJECT", "neural-controllers"),
    name=f"poetry-transfer-{os.environ.get('SLURM_JOB_ID', 'local')}",
    config={
        "model": "Llama-3.1-8B-Instruct",
        "control_method": "rfm",
        "rfm_iters": 8,
        "batch_size": 2,
        "n_components": 5
    }
)

# Setup paths
base_dir = '/work/hdd/bdau/mbanisharifdehkordi/neural_controllers'
sys.path.insert(0, base_dir)
os.environ['HF_HOME'] = os.path.join(base_dir, 'model_cache')

from neural_controllers import NeuralController

# Load model and tokenizer
model_path = os.path.join(base_dir, 'models/Llama-3.1-8B-Instruct')
print("Loading model and tokenizer...")
wandb.log({"status": "loading_model"})

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("Model loaded successfully!")
wandb.log({"status": "model_loaded"})

# Create neural controller
controller = NeuralController(
    model,
    tokenizer,
    rfm_iters=8,
    batch_size=2,
    n_components=5,
    control_method='rfm'
)

# Load poetry data
poetry_dir = os.path.join(base_dir, 'data/poetry')
with open(os.path.join(poetry_dir, 'sentences.txt'), 'r') as f:
    normal_sentences = f.readlines()[:5]
with open(os.path.join(poetry_dir, 'poems.txt'), 'r') as f:
    poetry_sentences = f.readlines()[:5]

# Prepare training data
train_inputs = []
train_labels = []

for sentence in normal_sentences:
    train_inputs.append(controller.format_prompt(sentence.strip()))
    train_labels.append(0)
for poem in poetry_sentences:
    train_inputs.append(controller.format_prompt(poem.strip()))
    train_labels.append(1)

wandb.log({"n_train_samples": len(train_inputs)})

# Compute directions
print("Computing poetry style directions...")
controller.compute_directions(train_inputs, train_labels)
wandb.log({"status": "directions_computed"})

# Test generation
test_prompt = "What can I do to treat flu symptoms?"
formatted_prompt = controller.format_prompt(test_prompt)

# Generate outputs
normal_output = controller.generate(
    formatted_prompt,
    layers_to_control=[],
    control_coef=0,
    max_new_tokens=50
)

poetic_output = controller.generate(
    formatted_prompt,
    layers_to_control=list(range(-1, -15, -1)),
    control_coef=0.5,
    max_new_tokens=50
)

# Log results to wandb
wandb.log({
    "test_prompt": test_prompt,
    "normal_output": normal_output,
    "poetic_output": poetic_output
})

print(f"\nNormal: {normal_output}")
print(f"\nPoetic: {poetic_output}")

wandb.finish()
print("\n✓ Experiment logged to wandb!")