import sys
import optimizers

TASK_PROMPT_TEMPLATE = (
    "You must minimize the blackbox function F.\n"
    "F is a function of {n_dimensions} parameters and returns a real number.\n"
    "It has a single global minimum.\n"
    "The parameters of F and their domains are:\n"
    "{parameter_space_repr}\n"
    "\n"
)

parameter_space_repr = "x1 (float): [-5, 5]\nx2 (float): [-5, 5]"

if __name__ == '__main__':

    optimizer_name = sys.argv[1]
    optimizer = getattr(optimizers, optimizer_name)

    function = lambda x: (x[0] - 1.25)**2 + (x[1] - 2.3)**2
    task_prompt = TASK_PROMPT_TEMPLATE.format(
        n_dimensions=2, 
        parameter_space_repr=parameter_space_repr
    )

    history, loop_times = optimizer(function, task_prompt, budget=20)
    print(history)
    print(loop_times)