from ._inference_auth_token import get_access_token
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

def get_llm(model="openai/gpt-oss-120b"):
    api_key = get_access_token()
    base_url = "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
    llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url)
    return llm

def get_agent(model, tools):
    api_key = get_access_token()
    base_url = "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
    llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url)
    agent = create_react_agent(llm, tools)
    return agent

def parameter_space_to_string(parameter_space):
    return "\n".join(
        f"{name} ({data_type.__name__}): {domain}"
        for name, (data_type, domain) in parameter_space.items()
    )

def history_to_string(history):
    return "\n".join([
        f"F({parameters.__repr__()}) = {value:.4f}" 
        for parameters, value in history
    ])