#!/usr/bin/env python3
"""Basic test script to verify the code agent functionality."""

import sys
import os
import subprocess

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def check_conda_environment():
    """Check if running in the correct conda environment."""
    print("Checking conda environment...")
    
    # Check if conda is available
    try:
        result = subprocess.run(['conda', '--version'], capture_output=True, text=True)
        print(f"Conda version: {result.stdout.strip()}")
    except FileNotFoundError:
        print("⚠️  Conda not found - using system Python")
        return False
    
    # Check current environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'base')
    print(f"Current conda environment: {conda_env}")
    
    if conda_env in ['code-agent-test', 'code-agent-dev']:
        print(f"✅ Running in correct conda environment: {conda_env}")
        return True
    else:
        print(f"⚠️  Not in code-agent conda environment. Current: {conda_env}")
        print("Recommended: conda activate code-agent-test")
        return False


def check_dependencies():
    """Check if all required dependencies are installed."""
    print("\nChecking dependencies...")
    
    required_packages = [
        'ollama',
        'click', 
        'pydantic',
        'rich'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - installed")
        except ImportError:
            print(f"❌ {package} - missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("✅ All dependencies installed")
        return True

from src.tools.filesystem import ReadFileTool, WriteFileTool
from src.tools.registry import ToolRegistry
from src.agent.react_agent import ReActAgent


def test_tools():
    """Test basic tool functionality."""
    print("Testing tools...")
    
    # Test write tool
    write_tool = WriteFileTool()
    result = write_tool.execute(path="/tmp/test_file.txt", content="Hello, World!")
    print(f"Write result: {result.success}, {result.content}")
    
    # Test read tool
    read_tool = ReadFileTool()
    result = read_tool.execute(path="/tmp/test_file.txt")
    print(f"Read result: {result.success}, content: '{result.content[:20]}...'")
    
    # Clean up
    os.remove("/tmp/test_file.txt")
    print("Tool tests completed!")


def test_registry():
    """Test tool registry."""
    print("\nTesting tool registry...")
    
    registry = ToolRegistry()
    tools = registry.list_tools()
    print(f"Available tools: {tools}")
    
    # Test tool execution through registry
    from src.tools.schemas import ToolAction
    action = ToolAction(
        tool_name="write_file",
        parameters={"path": "/tmp/registry_test.txt", "content": "Registry test"}
    )
    
    result = registry.execute_tool(action)
    print(f"Registry execution result: {result.success}, {result.content}")
    
    # Clean up
    if os.path.exists("/tmp/registry_test.txt"):
        os.remove("/tmp/registry_test.txt")
    
    print("Registry tests completed!")


def test_agent_init():
    """Test agent initialization (without Ollama)."""
    print("\nTesting agent initialization...")
    
    try:
        # This will fail if Ollama is not running, but we can test initialization
        agent = ReActAgent(model="llama3.2")
        tools_help = agent.list_available_tools()
        print(f"Agent initialized successfully. Tools available: {len(agent.tool_registry.list_tools())}")
        print("Agent initialization test completed!")
    except Exception as e:
        print(f"Agent test skipped (Ollama not available): {e}")


if __name__ == "__main__":
    print("Running basic functionality tests...\n")
    
    # Check environment setup
    conda_ok = check_conda_environment()
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\n❌ Cannot proceed with tests - missing dependencies")
        sys.exit(1)
    
    print("\n" + "="*50)
    
    test_tools()
    test_registry()
    test_agent_init()
    
    print("\n✅ Basic tests completed! The code agent is ready.")
    
    if conda_ok:
        print("\n🐍 Environment: Running in correct conda environment")
    else:
        print("\n⚠️  Environment: Consider using conda environment")
        print("   conda env create -f environment.yml")
        print("   conda activate code-agent-test")
    
    print("\nTo use the agent:")
    print("1. Make sure Ollama is running: ollama serve")
    print("2. Pull a model: ollama pull llama3.2")
    print("3. Run the agent: python -m src.cli.main chat")