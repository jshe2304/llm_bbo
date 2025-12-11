import sys
import math
import json
import os
from datetime import datetime
from yahpo_gym.benchmark_set import BenchmarkSet
from tqdm import tqdm
import optimizers

TASK_PROMPT_TEMPLATE = (
    "You must find the hyperparameters which minimize the validation cross-entropy of a neural network.\n"
    "The model is a feedforward neural network trained on a OpenML dataset.\n"
    "There are {n_dimensions} hyperparameters to optimize, their domains are:\n"
    "{parameter_space_repr}\n"
    "\n"
)

# LCBench scenarios to benchmark on
EXCLUDED_PARAMS = {"OpenML_task_id"}

def format_bound(value, direction, sig_figs=1):
    """Format a numeric bound, rounding conservatively (up for lower, down for upper)."""
    if value == 0:
        return "0"
    if isinstance(value, int) or value == int(value):
        return str(int(value))
    
    # Round to sig_figs significant figures in the specified direction
    magnitude = math.floor(math.log10(abs(value)))
    scale = 10 ** (magnitude - sig_figs + 1)
    rounding_fn = math.ceil if direction == "up" else math.floor
    rounded = rounding_fn(value / scale) * scale
    
    return f"{rounded:.{sig_figs}g}"


def get_parameter_space_repr(bench):
    """Build a string representation of the parameter space for the LLM prompt."""
    config_space = bench.get_opt_space()
    lines = []
    
    for param_name in config_space.get_hyperparameter_names():
        # Skip excluded parameters (like OpenML_task_id which is fixed by instance)
        if param_name in EXCLUDED_PARAMS:
            continue
            
        param = config_space.get_hyperparameter(param_name)
        param_type = type(param).__name__
        
        if "Integer" in param_type:
            lines.append(f"{param_name} (int): [{param.lower}, {param.upper}]")
        elif "Float" in param_type or "Uniform" in param_type:
            lower = format_bound(param.lower, "up")
            upper = format_bound(param.upper, "down")
            lines.append(f"{param_name} (float): [{lower}, {upper}]")
        elif "Categorical" in param_type:
            lines.append(f"{param_name} (categorical): {list(param.choices)}")
        else:
            # Fallback for other types
            if hasattr(param, "bounds"):
                lines.append(f"{param_name}: {param.bounds}")
            elif hasattr(param, "choices"):
                lines.append(f"{param_name}: {list(param.choices)}")
    
    return '\n'.join(lines)


def get_param_names(bench):
    """Get ordered list of parameter names from the optimization space (excluding fixed params)."""
    config_space = bench.get_opt_space()
    return [name for name in config_space.get_hyperparameter_names() if name not in EXCLUDED_PARAMS]


def create_objective_wrapper(bench, param_names, instance_id):
    """
    Create a wrapper function that converts tuple inputs to dictionary configs
    and returns the target metric value.
    """
    def objective(params_tuple):
        # Convert tuple to dictionary
        config = {name: val for name, val in zip(param_names, params_tuple)}
        
        # Add the fixed instance parameter
        config["OpenML_task_id"] = instance_id
        
        # Evaluate on YAHPO
        result = bench.objective_function(config)
        
        # Return the target metric (first result if list)
        if isinstance(result, list):
            result = result[0]
        
        return float(result['val_cross_entropy'])
    
    return objective


def save_history(history, output_path):
    """
    Save optimization history to a JSON file.
    
    Args:
        history: List of (candidate_tuple, function_value, loop_time) tuples
        output_path: Path to save the history
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Format history entries with named parameters

    # Save to file
    with open(output_path, 'w') as f:
        json.dump(history, f, indent=2)

def benchmark(optimizer, budget=32):
    """Run the optimizer on YAHPO LCBench instances."""

    bench = BenchmarkSet("lcbench")
    
    for instance_id in tqdm(bench.instances, desc="Instances"):

        # Set benchmark instance
        bench.set_instance(instance_id)
        
        # Get parameter info
        param_names = get_param_names(bench)
        
        # Build task prompt
        task_prompt = TASK_PROMPT_TEMPLATE.format(
            n_dimensions=len(param_names),
            parameter_space_repr=get_parameter_space_repr(bench)
        )

        # Create objective wrapper
        objective = create_objective_wrapper(bench, param_names, instance_id)
        
        # Run optimizer
        history = optimizer(objective, task_prompt, budget=budget)
        
        # Save history to file
        results_path = os.path.join("./yahpo_results", f"long_{optimizer.__name__}", f"{instance_id}.json")
        save_history(history, results_path)

if __name__ == "__main__":
    optimizer_name = sys.argv[1]
    optimizer = getattr(optimizers, optimizer_name)

    benchmark(optimizer)
    