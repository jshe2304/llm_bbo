import sys
import cocoex  # pip install cocopp
from tqdm import tqdm
import optimizers

TASK_PROMPT_TEMPLATE = (
    "You must minimize the blackbox function F.\n"
    "F is a function of {n_dimensions} parameters and returns a real number.\n"
    "It has a single global minimum.\n"
    "The parameters of F and their domains are:\n"
    "{parameter_space_repr}\n"
    "\n"
)

def get_parameter_space_repr(problem):

    lb = problem.lower_bounds
    ub = problem.upper_bounds

    return '\n'.join([
        f'x{i} (float): [{float(lb[i])}, {float(ub[i])}]'
         for i in range(problem.dimension)
    ])

def benchmark(optimizer):

    # Create a fresh suite for this optimizer
    suite_name = "bbob"
    suite_options = "dimensions: 2 instances: 1" 

    # Create a fresh suite for this optimizer
    suite = cocoex.Suite(suite_name, "", suite_options)

    # Create result folder
    observer = cocoex.Observer(
        suite_name,
        f"result_folder: {optimizer.__name__}_on_{suite_name} "
        f"algorithm_name: {optimizer.__name__}"
    )

    # Loop over all problems in the suite
    for problem in tqdm(suite):
        problem.observe_with(observer)

        # Build task prompt
        task_prompt = TASK_PROMPT_TEMPLATE.format(
            n_dimensions=problem.dimension,
            parameter_space_repr=get_parameter_space_repr(problem)
        )

        best_parameters, best_value = optimizer(problem, task_prompt, budget=20)
        print(f"Best parameters: {best_parameters}")
        print(f"Best value: {best_value}")
        break

if __name__ == "__main__":
    optimizer_name = sys.argv[1]
    optimizer = getattr(optimizers, optimizer_name)
    benchmark(optimizer)