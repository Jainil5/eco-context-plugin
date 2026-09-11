import os
import subprocess
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain.prompts import PromptTemplate

def execute_shell(command: str) -> str:
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
        output = result.stdout
        if result.stderr:
            output += "\nSTDERR:\n" + result.stderr
        return output[:4000] # Limit output length
    except Exception as e:
        return f"Error executing command: {e}"

def read_file(filepath: str) -> str:
    try:
        with open(filepath, 'r') as f:
            return f.read()[:8000]
    except Exception as e:
        return f"Error reading file: {e}"

def write_file(args: str) -> str:
    # args format: filepath|content
    try:
        parts = args.split("|", 1)
        if len(parts) != 2:
            return "Error: arguments must be in the format 'filepath|content'"
        filepath, content = parts
        with open(filepath.strip(), 'w') as f:
            f.write(content)
        return f"Successfully written to {filepath}"
    except Exception as e:
        return f"Error writing file: {e}"

def create_agent(model_name="gpt-4o-mini", api_key=None):
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY", "dummy_key")
    
    llm = ChatOpenAI(model=model_name, api_key=api_key, temperature=0)

    tools = [
        Tool(
            name="ExecuteShell",
            func=execute_shell,
            description="Executes a shell command. Useful for running tests, listing files (ls), searching (grep). Returns stdout/stderr."
        ),
        Tool(
            name="ReadFile",
            func=read_file,
            description="Reads the content of a file. Provide the filepath as the argument."
        ),
        Tool(
            name="WriteFile",
            func=write_file,
            description="Writes content to a file. Argument must be exactly 'filepath|content'."
        )
    ]

    template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

    prompt = PromptTemplate.from_template(template)
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=15)
    return agent_executor
