from .utils import get_llm
import time

def history_to_string(history):
    if len(history) == 0: return ""

    history_str = "Your guesses and their evaluated values so far:\n"
    for parameters, value, _ in history:
        history_str += f"F({parameters.__repr__()}) = {value:.4f}\n"
    history_str += "\n"
    
    return history_str

GUESS_PROMPT = (
    "Task:\n"
    "Make an educated guess at a new parameter set that will further minimize F. \n"
    "Your evaluation budget is limited, so use them wisely. \n"
    "\n"
    "Constraints:\n"
    "Your response should be a tuple that can be evaluated in python. \n" 
    "It should have the format \"(x1, ..., x_n)\", where the x_i are the parameters in the order specified in the task description. \n"
    "The parameters must be within the correct domains. \n"
    "Do not include any other text in your response. \n"
)

def llm_optimizer(function, task_prompt: str, budget: int = 32):

    llm = get_llm("openai/gpt-oss-120b")
    history = []
    loop_times = []

    this_budget = budget
    while this_budget > 0:
        loop_start = time.time()

        # Construct prompt
        prompt = task_prompt + history_to_string(history) + GUESS_PROMPT

        # Invoke LLM response
        raw = llm.invoke(prompt).content.strip()

        # Parse response
        try: 
            candidate = eval(raw)
            value = float(function(candidate))
        except:
            continue

        # Evaluate candidate and update history
        loop_time = time.time() - loop_start
        history.append((candidate, value, loop_time))

        this_budget -= 1

    # Return history
    return history