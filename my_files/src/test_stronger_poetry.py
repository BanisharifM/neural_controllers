import os
import sys
import torch
import wandb

base_dir = '/work/hdd/bdau/mbanisharifdehkordi/neural_controllers'
sys.path.insert(0, base_dir)
os.environ['HF_HOME'] = os.path.join(base_dir, 'model_cache')

from transformers import AutoTokenizer, AutoModelForCausalLM
from neural_controllers import NeuralController

# Initialize wandb
wandb.init(
    project=os.environ.get("WANDB_PROJECT", "neural-controllers"),
    name=f"poetry-test-coefficients-{os.environ.get('SLURM_JOB_ID', 'local')}",
    config={
        "test": "coefficient_sweep",
        "model": "Llama-3.1-8B",
        "control_method": "rfm"
    }
)

# Load model
model_path = os.path.join(base_dir, 'models/Llama-3.1-8B-Instruct')
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="cuda"
)

# Create controller and load saved directions
controller = NeuralController(model, tokenizer, control_method='rfm')
controller.load(concept='poetry', model_name='llama_3_8b_it', 
                path=os.path.join(base_dir, 'my_files/directions/'))

# Test with different coefficients
prompt = "Tell me about the weather"
formatted = controller.format_prompt(prompt)

results = []
for coef in [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
    output = controller.generate(
        formatted,
        layers_to_control=list(range(-1, -31, -1)) if coef > 0 else [],
        control_coef=coef,
        max_new_tokens=100,
        do_sample=False
    ).replace(formatted, "").replace("<|start_header_id|>assistant<|end_header_id|>", "").strip()
    
    results.append([coef, output])
    
    # Log each result
    wandb.log({
        f"coefficient_{coef}": {
            "output": output[:1000],  # First 1000 chars
            "control_strength": coef
        }
    })

# Create comparison table
wandb.log({
    "coefficient_comparison": wandb.Table(
        columns=["coefficient", "output"],
        data=results
    )
})

wandb.finish()
print("\n✓ Test with wandb logging completed")