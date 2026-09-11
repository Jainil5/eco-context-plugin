import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
import traceback

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from loader import SWEBenchLoader
from agent import create_agent

def clone_repo(repo_name, base_commit, target_dir):
    print(f"Cloning {repo_name} to {target_dir} at commit {base_commit}...")
    if not os.path.exists(target_dir):
        subprocess.run(["git", "clone", f"https://github.com/{repo_name}.git", target_dir], check=True)
    subprocess.run(["git", "checkout", base_commit], cwd=target_dir, check=True)

def apply_test_patch(target_dir, test_patch):
    if test_patch and test_patch.strip():
        patch_path = os.path.join(target_dir, "test_patch.diff")
        with open(patch_path, "w") as f:
            f.write(test_patch)
        try:
            subprocess.run(["git", "apply", "test_patch.diff"], cwd=target_dir, check=True)
            print("Successfully applied test patch.")
        except subprocess.CalledProcessError:
            print("Warning: Failed to apply test patch cleanly. Proceeding anyway...")

def run_tests(target_dir):
    """A very simple baseline test runner using pytest."""
    try:
        # Many python projects use pytest
        result = subprocess.run(["pytest"], cwd=target_dir, capture_output=True, text=True, timeout=60)
        passed = result.returncode == 0
        return passed, result.stdout + "\n" + result.stderr
    except Exception as e:
        return False, str(e)

def run_baseline(max_tasks=3):
    loader = SWEBenchLoader()
    tasks = loader.load_tasks(limit=max_tasks)
    
    agent = create_agent()
    
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    for task in tasks:
        task_id = task.get("instance_id", "unknown_task")
        print(f"\n--- Processing Task: {task_id} ---")
        
        repo_name = task["repo"]
        base_commit = task["base_commit"]
        problem_statement = task["problem_statement"]
        
        workspace_dir = os.path.abspath(os.path.join("dataset", "workspaces", task_id))
        os.makedirs(workspace_dir, exist_ok=True)
        
        try:
            clone_repo(repo_name, base_commit, workspace_dir)
            apply_test_patch(workspace_dir, task.get("test_patch", ""))
            
            # Formulate prompt for the agent
            prompt = (
                f"You are a coding agent. The user has provided the following problem statement:\n\n"
                f"{problem_statement}\n\n"
                f"The repository is located at: {workspace_dir}\n"
                f"Please navigate to the directory using ExecuteShell, examine the codebase, and write the necessary fixes.\n"
                f"When you are done, describe the changes you made."
            )
            
            print("Invoking LangChain Agent...")
            start_time = time.time()
            
            # Since this is a test and we don't have infinite context or time, we just run the agent
            # We assume OPENAI_API_KEY is exported or the user provides it.
            # For the baseline run, if there's no API key, it will fail gracefully.
            try:
                response = agent.invoke({"input": prompt})
            except Exception as e:
                print(f"Agent failed: {e}")
                traceback.print_exc()
            
            # Capture patch
            diff_result = subprocess.run(["git", "diff"], cwd=workspace_dir, capture_output=True, text=True)
            generated_patch = diff_result.stdout
            
            duration = time.time() - start_time
            
            print("Running tests...")
            tests_passed, test_logs = run_tests(workspace_dir)
            
            # Save results
            result_data = {
                "task_id": task_id,
                "repository": repo_name,
                "base_commit": base_commit,
                "success": bool(generated_patch.strip()) and tests_passed,
                "tests_passed": tests_passed,
                "duration": duration,
                "changed_files": [] # Could parse diff to get this
            }
            
            with open(os.path.join(results_dir, f"{task_id}.json"), "w") as f:
                json.dump(result_data, f, indent=2)
                
            with open(os.path.join(results_dir, f"{task_id}_test.log"), "w") as f:
                f.write(test_logs)
                
            with open(os.path.join(results_dir, f"{task_id}.patch"), "w") as f:
                f.write(generated_patch)
                
            print(f"Completed {task_id}. Passed: {tests_passed}")
            
        except Exception as e:
            print(f"Error processing {task_id}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tasks", type=int, default=3, help="Max tasks to run")
    args = parser.parse_args()
    run_baseline(max_tasks=args.max_tasks)
