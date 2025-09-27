import os
import sys
import torch
import wandb
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add path for imports
base_dir = '/work/hdd/bdau/mbanisharifdehkordi/neural_controllers'
sys.path.insert(0, base_dir)

from neural_controllers import NeuralController
import utils

# Set environment variables
os.environ['HF_HOME'] = os.path.join(base_dir, 'model_cache')

# Initialize wandb
wandb.init(
    project=os.environ.get("WANDB_PROJECT", "neural-controllers"),
    name=f"poetry-style-{os.environ.get('SLURM_JOB_ID', 'local')}",
    config={
        "model": "Llama-3.1-8B",
        "control_method": "rfm",
        "task": "poetry_style_transfer",
        "rfm_iters": 8,
        "batch_size": 2
    }
)

print("="*60)
print("Poetry Style Transfer - From Notebook (with wandb)")
print("="*60)

# Model configuration
model_type = 'llama'

if model_type == 'llama':
    model_path = os.path.join(base_dir, 'models/Llama-3.1-8B-Instruct')
    
    print("Loading Llama model...")
    wandb.log({"status": "loading_model"})
    
    language_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", legacy=False)
    model_name = 'llama_3_8b_it'
    assistant_tag = '<|start_header_id|>assistant<|end_header_id|>'

tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id

# Load poetry dataset
data_dir = os.path.join(base_dir, "data/poetry")
print(f"\nLoading data from: {data_dir}")
dataset = utils.poetry_dataset(data_dir=data_dir, tokenizer=tokenizer, assistant_tag=assistant_tag)

wandb.log({"status": "dataset_loaded"})

# Train controllers
concept_types = ['prose', 'poetry']
controllers = {}

print("\nTraining controllers...")
for concept_type in tqdm(concept_types, desc="Training"):
    train_data = dataset[concept_type]['train']
    
    wandb.log({
        f"{concept_type}_train_size": len(train_data['inputs']),
        "current_concept": concept_type
    })
    
    controller = NeuralController(
        language_model,
        tokenizer,
        rfm_iters=8,
        batch_size=2,
        control_method='rfm'
    )
    
    print(f"\n  Computing directions for {concept_type}...")
    controller.compute_directions(train_data['inputs'], train_data['labels'])
    
    controllers[concept_type] = controller
    wandb.log({f"{concept_type}_directions_computed": True})

# Save directions
directions_path = os.path.join(base_dir, 'my_files/directions')
os.makedirs(directions_path, exist_ok=True)

for concept_type in concept_types:
    controller = controllers[concept_type]
    controller.save(concept=f'{concept_type}', model_name=model_name, path=directions_path)
    wandb.log({f"{concept_type}_directions_saved": True})

# Test the controllers
print("\n" + "="*60)
print("Testing Style Transfer")
print("="*60)

concept_type = "poetry"
controller = controllers[concept_type]

raw_inputs = [
    "What can I buy in a grocery store?",
    "How should I treat a cold?",
    "Tell me about something interesting.",
]

inputs = [controller.format_prompt(x) for x in raw_inputs]

num_new_tokens = 100
coef = 0.7
layers = list(range(-1, -31, -1))

wandb.log({
    "test_concept": concept_type,
    "control_coefficient": coef,
    "num_test_prompts": len(raw_inputs),
    "max_new_tokens": num_new_tokens
})

print(f"\nApplying {concept_type} style with coefficient {coef}")

# Store outputs for wandb
test_results = []

for idx, i in enumerate(inputs):
    print(f"\n[{idx+1}] Original prompt: {raw_inputs[idx]}")
    
    gen = controller.generate(
        i, 
        layers_to_control=layers, 
        control_coef=coef,
        max_new_tokens=num_new_tokens, 
        do_sample=False
    ).replace(i, "")
    
    print(f"Response: {gen[:200]}...")  # Print first 200 chars
    
    # Log to wandb
    test_results.append({
        "prompt": raw_inputs[idx],
        "response": gen
    })

# Create wandb table with results
wandb.log({
    "test_results": wandb.Table(
        columns=["prompt", "response"],
        data=[[r["prompt"], r["response"]] for r in test_results]
    )
})

wandb.log({"status": "completed"})
wandb.finish()

print("\n✓ Poetry notebook with wandb logging completed!") 
