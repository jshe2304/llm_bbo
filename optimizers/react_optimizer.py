from .utils import get_llm

import time
from typing import List, Dict, Any
from typing_extensions import TypedDict, Annotated

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

REASON_PROMPT_TEMPLATE = (
    "Reason about the next point you should evaluate F at.\n"
    "You may want to think about the following: \n"
    "1. What do we know already about the structure of F?\n"
    "2. What information do we need to gather?\n"
    "3. Where in the parameter space should we gather that information?\n"
    "Remember, you only have {remaining_budget} evaluations remaining,\n"
    "so you should think carefully about where to use your next evaluation.\n"
)

REASON_PROMPT_TEMPLATE = (
    "Reason about the next point you should evaluate F at.\n"
    "You may want to think about the following: \n"
    "1. What do we know already about the structure of F?\n"
    "2. What information do we need to gather?\n"
    "3. Where in the parameter space should we gather that information?\n"
    "Remember, you only have {remaining_budget} evaluations remaining,\n"
    "so you should think carefully about where to use your next evaluation.\n"
)

ACT_PROMPT_TEMPLATE = (
    "You are an optimizer minimizing a function F.\n"
    "You have already reasoned about F and where you should evaluate it next.\n"
    "Based on your reasoning, call the tool to evaluate F at a point .\n"
)

class AgentState(TypedDict):
    task_description: BaseMessage
    evaluation_history: List[BaseMessage]
    reasoning: BaseMessage
    remaining_budget: int
    
class ReactOptimizer:
    def __init__(self, function, budget: int = 20):
        self.function = function
        self.budget = budget

        # History of evaluations
        self.history = []

        # Create tool(s) bound to this instance without exposing `self` in schema
        @tool
        def _evaluate_function(x: tuple) -> float:
            '''
            Evaluate the function F at the point specified by the tuple x.
            - x must be a Python tuple/list of the correct dimensionality.
            - The parameters in x must be in the order described.
            - The parameters in x must be within the correct domains.
            Returns: the scalar value F(x).
            '''
            value = function(x)
            self.history.append((x, value))
            return value

        # Create hidden tool node
        tools = [_evaluate_function]
        self._tool_node = ToolNode(tools)

        # Create LLMs
        self.reasoning_llm = get_llm("openai/gpt-oss-120b")
        self.acting_llm = get_llm("openai/gpt-oss-20b").bind_tools(tools)

    def tool_node(self, state: AgentState) -> AgentState:
        '''
        Tool node wrapper that updates the budget
        '''
        tool_result = self._tool_node.invoke({"messages": [state["evaluation_history"][-1]]})
        state["evaluation_history"].extend(tool_result["messages"])
        state["remaining_budget"] -= 1
        return state

    def reason_node(self, state: AgentState) -> AgentState:
        '''
        Reasoning node which prompts the LLM to reason about the next point to evaluate. 
        The model is shown:
         1. task description
         2. past evaluations and their results
         3. prompt to reason about the next point to evaluate
        '''

        # Format prompt
        prompt = REASON_PROMPT_TEMPLATE.format(
            remaining_budget=state["remaining_budget"],
        )

        # Build model context
        msgs = [
            state["task_description"],
            *state["evaluation_history"],
            SystemMessage(content=prompt),
        ]

        # Invoke model
        resp = self.reasoning_llm.invoke(msgs)

        # Update state
        state['reasoning'] = resp
        return state

    def act_node(self, state: AgentState) -> AgentState:
        '''
        Action node which prompts the LLM to call the evaluation tool. 
        The model is shown:
         1. task description
         2. past evaluations and their results
         3. reasoning from the previous node
         4. prompt to call the tool
        '''

        # Construct model context
        msgs = [
            state["task_description"],
            *state["evaluation_history"],
            state["reasoning"], 
            SystemMessage(content=ACT_PROMPT_TEMPLATE),
        ]

        # Invoke model
        resp = self.acting_llm.invoke(msgs)

        # Update state
        state['evaluation_history'].append(resp)
        return state

    def budget_exhausted(self, state: AgentState) -> str:
        return "reason" if state["remaining_budget"] > 0 else END

    def build_graph(self):
        
        # Define graph nodes
        workflow = StateGraph(AgentState)
        workflow.add_node("reason", self.reason_node)
        workflow.add_node("act", self.act_node)
        workflow.add_node("tools", self.tool_node)

        # Define connectivity
        workflow.set_entry_point("reason")
        workflow.add_edge("reason", "act")
        workflow.add_edge("act", "tools")
        workflow.add_conditional_edges(
            "tools",
            self.budget_exhausted,
            {
                "reason": "reason",
                END: END,
            },
        )

        # Compile graph
        agent = workflow.compile()

        return agent

def print_message(message, use_color: bool = True):
    """Pretty-print a message from the optimizer agent stream."""
    
    # ANSI color codes
    COLORS = {
        'reason': '\033[94m',   # Blue
        'act': '\033[93m',      # Yellow
        'tools': '\033[92m',    # Green
        'reset': '\033[0m',
        'bold': '\033[1m',
        'dim': '\033[2m',
    } if use_color else {k: '' for k in ['reason', 'act', 'tools', 'reset', 'bold', 'dim']}
    
    def header(name: str, color_key: str):
        color = COLORS[color_key]
        reset = COLORS['reset']
        bold = COLORS['bold']
        return f"{color}{bold}{'─' * 3} {name} {'─' * (40 - len(name))}{reset}"
    
    try:
        if 'reason' in message:
            state = message['reason']
            print(header("Reason", "reason"))
            reasoning = state.get('reasoning')
            if reasoning and hasattr(reasoning, 'content'):
                print(reasoning.content)
            budget = state.get('remaining_budget')
            if budget is not None:
                print(f"{COLORS['dim']}[Budget remaining: {budget}]{COLORS['reset']}")
                
        elif 'act' in message:
            state = message['act']
            print(header("Act", "act"))
            eval_history = state.get('evaluation_history', [])
            if eval_history:
                last_msg = eval_history[-1]
                tool_calls = getattr(last_msg, 'additional_kwargs', {}).get('tool_calls', [])
                if tool_calls:
                    for tc in tool_calls:
                        func = tc.get('function', {})
                        name = func.get('name', 'unknown')
                        args = func.get('arguments', '{}')
                        print(f"  → {name}({args})")
                else:
                    print(f"  {last_msg.content if hasattr(last_msg, 'content') else last_msg}")
                    
        elif 'tools' in message:
            state = message['tools']
            print(header("Tools", "tools"))
            eval_history = state.get('evaluation_history', [])
            if eval_history:
                # Show the tool result (usually the last message)
                last_msg = eval_history[-1]
                if hasattr(last_msg, 'content'):
                    print(f"  Result: {last_msg.content}")
                else:
                    print(f"  {last_msg}")
            budget = state.get('remaining_budget')
            if budget is not None:
                print(f"{COLORS['dim']}[Budget remaining: {budget}]{COLORS['reset']}")
        else:
            # Unknown message type - show keys
            print(f"[Unknown node: {list(message.keys())}]")
            
    except (KeyError, AttributeError, IndexError, TypeError) as e:
        # Fallback with error context for debugging
        print(f"[Parse error: {type(e).__name__}] {message}")
    
    print()

def react_optimizer(function, task_prompt: str, budget: int = 32): 
    '''
    Function wrapper to run the ReactOptimizer agent on a problem.
    '''
    # Build graph
    agent = ReactOptimizer(function, budget)
    optimizer = agent.build_graph()

    # Run optimizer agent
    prompt = task_prompt
    prompt += f"\nYou only have {budget} evaluations in total, use them wisely.\n"
    
    # Track timing for each loop iteration
    loop_times = []
    loop_start = None
    
    # Stream optimizer agent to time each loop
    for message in optimizer.stream(
        {
            "task_description": SystemMessage(content=prompt),
            "evaluation_history": [],
            "remaining_budget": budget,
            "reasoning": None,
        },
        {'recursion_limit': 3 * budget + 1}
    ):
        print_message(message)
        # Start timing at reason node (beginning of loop)
        if 'reason' in message:
            loop_start = time.time()
        
        # End timing at tools node (end of loop)
        if 'tools' in message and loop_start is not None:
            loop_times.append(time.time() - loop_start)
            loop_start = None

    # Return history
    return [(x, v, t) for (x, v), t in zip(agent.history, loop_times)]
