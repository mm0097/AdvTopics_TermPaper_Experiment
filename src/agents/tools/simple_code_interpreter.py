"""
Simplified code interpreter that executes Python in Docker without Jupyter complexity.
More reliable for agent usage.
"""

import docker
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from langchain_core.tools import BaseTool
from pydantic import Field, BaseModel


@dataclass
class CodeExecutionResult:
    """Result of code execution."""
    success: bool
    output: str
    error: Optional[str] = None
    execution_time: float = 0.0


class SimpleCodeInterpreterSchema(BaseModel):
    """Schema for simple code interpreter tool arguments."""
    code: str = Field(description="Python code to execute")


class SimpleCodeInterpreterTool(BaseTool):
    """
    Simple Python code execution environment using Docker.
    
    This tool provides secure, sandboxed Python execution with scientific libraries.
    """
    
    name: str = "code_interpreter"
    description: str = """Secure Python code execution environment in Docker container.
    
    Capabilities:
    - Mathematical calculations and statistical analysis
    - Data processing with NumPy, Pandas
    - Visualization with Matplotlib (saved to files)
    - Scientific computing with SciPy, scikit-learn
    
    Input: Python code as a string
    Output: Execution results including output and any errors
    
    The environment includes: numpy, pandas, matplotlib, seaborn, scipy, scikit-learn"""
    
    timeout: float = Field(default=60.0)
    docker_client: Optional[docker.DockerClient] = Field(default=None, exclude=True)
    
    def __init__(self, timeout: float = 60.0, **kwargs):
        super().__init__(**kwargs)
        self.timeout = timeout
        self.docker_client = None
        # Set args_schema as class, not instance - this is the correct way for LangChain tools
        self.args_schema = SimpleCodeInterpreterSchema
    
    def _run(self, code: str, **kwargs) -> str:
        """Execute Python code in Docker container."""
        start_time = time.time()
        
        try:
            # Initialize Docker client
            if self.docker_client is None:
                self.docker_client = docker.from_env()
            
            # Create temporary directory for code and outputs
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Modify code to capture last expression
                # Wrap the code to capture the last expression result
                wrapped_code = f"""
import sys
import traceback
from io import StringIO

# Capture both stdout and the final expression value
original_stdout = sys.stdout
captured_output = StringIO()
sys.stdout = captured_output

try:
    # Execute the user code
    exec_globals = {{'__name__': '__main__'}}
    
    # Split the code into lines and check if last line is an expression
    lines = '''{code}'''.strip().split('\\n')
    if lines:
        # Check if the last non-empty line is likely an expression
        last_line = lines[-1].strip()
        if last_line and not any(last_line.startswith(kw) for kw in ['import ', 'from ', 'def ', 'class ', 'if ', 'for ', 'while ', 'with ', 'try:', 'except', 'finally:', 'elif ', 'else:']):
            # Try to evaluate the last line as an expression
            code_without_last = '\\n'.join(lines[:-1])
            if code_without_last.strip():
                exec(code_without_last, exec_globals)
            try:
                result = eval(last_line, exec_globals)
                if result is not None:
                    print(result)
            except:
                # If evaluation fails, execute as statement
                exec(last_line, exec_globals)
        else:
            # Execute all as statements
            exec('''{code}''', exec_globals)
    
except Exception as e:
    print(f"Error: {{e}}")
    traceback.print_exc()

finally:
    sys.stdout = original_stdout
    print(captured_output.getvalue(), end='')
"""
                
                # Write wrapped code to file
                code_file = temp_path / "code_to_run.py"
                with open(code_file, 'w') as f:
                    f.write(wrapped_code)
                
                # Run code in Docker container
                container = self.docker_client.containers.run(
                    "code-interpreter-sandbox",
                    command=["python3", "/app/work/code_to_run.py"],
                    volumes={str(temp_path): {'bind': '/app/work', 'mode': 'rw'}},
                    working_dir="/app/work",
                    network_mode="none",  # Complete network isolation for security
                    auto_remove=False,
                    detach=True,
                    cpu_period=100000,
                    cpu_quota=50000,   # 0.5 CPU limit (down from 2.0)
                    mem_limit="256m"   # 256MB RAM limit (down from 1GB)
                )
                
                try:
                    # Wait for completion
                    exit_code = container.wait(timeout=self.timeout)
                    
                    # Get output
                    logs = container.logs().decode('utf-8')
                    
                    # Check exit code
                    if exit_code['StatusCode'] == 0:
                        execution_time = time.time() - start_time
                        return f"""Output:
{logs}

Execution time: {execution_time:.2f}s"""
                    else:
                        execution_time = time.time() - start_time
                        return f"""Code execution failed with exit code {exit_code['StatusCode']}:
{logs}

Execution time: {execution_time:.2f}s"""
                
                finally:
                    try:
                        container.remove()
                    except:
                        pass
                        
        except Exception as e:
            execution_time = time.time() - start_time
            return f"""Code execution failed: {str(e)}

Execution time: {execution_time:.2f}s"""
    
    async def _arun(self, code: str, **kwargs) -> str:
        """Async wrapper - just calls sync version."""
        return self._run(code, **kwargs)


def create_simple_code_interpreter_tool(timeout: float = 60.0) -> SimpleCodeInterpreterTool:
    """Create and return a simple code interpreter tool."""
    return SimpleCodeInterpreterTool(timeout=timeout)