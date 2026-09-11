# ECO - Engineering Code Optimization

An agent-based system designed to solve software engineering tasks by automatically generating code patches for real-world issues from the SWE-bench dataset.

## Project Overview

ECO (Engineering Code Optimization) is a framework that uses language model agents to automatically fix software bugs by:
1. Loading tasks from the SWE-bench Lite dataset
2. Creating isolated workspaces for each task
3. Running language model agents to analyze problem statements and generate code changes
4. Evaluating generated patches by applying them and running tests
5. Saving results including patches, test logs, and success metrics

## Repository Structure

```
eco/
├── dataset/                    # Dataset files and workspaces
│   └── swebench_cl/           # SWE-bench Lite dataset cache
├── scripts/                   # Main execution scripts
│   └── run_baseline.py        # Entry point for running baseline experiments
├── src/                       # Core source code
│   ├── agent.py              # LangChain agent implementation with shell/file tools
│   └── loader.py             # Dataset loading utility for SWE-bench
├── venv/                      # Python virtual environment
├── requirements.txt           # Python dependencies
└── results/                   # Generated results (created at runtime)
    ├── *.json                 # Task results metadata
    ├── *.patch                # Generated code patches
    └── *_test.log            # Test execution logs
```

## Installation

1. Clone this repository
2. Create and activate a virtual environment:
   ```bash
   python -m venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Usage

Run the baseline experiment with a specified number of tasks:

```bash
python scripts/run_baseline.py --max-tasks 5
```

Arguments:
- `--max-tasks`: Number of tasks to process from the dataset (default: 3)

## How It Works

1. **Task Loading**: The system loads tasks from the SWE-bench Lite dataset using `src/loader.py`
2. **Workspace Creation**: For each task, a dedicated workspace directory is created
3. **Repository Setup**: The agent clones the target repository at the specified base commit
4. **Agent Execution**: A LangChain ReAct agent is invoked with access to:
   - `ExecuteShell`: Run shell commands (git, pytest, etc.)
   - `ReadFile`: Read file contents
   - `WriteFile`: Write file contents
5. **Patch Generation**: The agent analyzes the problem statement and attempts to generate fixes
6. **Evaluation**: Generated patches are tested using pytest to determine success
7. **Results Storage**: All outputs are saved to the `results/` directory

## Agent Capabilities

The agent has access to three core tools:
- **ExecuteShell**: Run arbitrary shell commands with timeout protection
- **ReadFile**: Read contents of any file in the workspace
- **WriteFile**: Write content to files using `filepath|content` format

## Results Format

For each processed task, the system generates:
- `{task_id}.json`: Metadata including success status, duration, and repository info
- `{task_id}.patch`: The generated code diff/patch
- `{task_id}_test.log`: Complete test output from pytest execution

## Dependencies

See `requirements.txt` for the complete list:
- langchain
- langchain-openai
- datasets
- pandas
- GitPython

## Notes

- The system assumes an OpenAI API key is available via the `OPENAI_API_KEY` environment variable
- Test execution uses a simple pytest runner with a 60-second timeout
- Workspace directories are preserved after execution for inspection
- If test patches fail to apply cleanly, the system continues execution anyway

## License

MIT