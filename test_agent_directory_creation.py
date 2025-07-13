#!/usr/bin/env python3
"""Test script for agent directory structure creation using tools."""

import sys
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.tools.filesystem import CreateDirectoryTool, WriteFileTool, ListDirectoryTool
from src.tools.registry import ToolRegistry
from src.tools.schemas import ToolAction, ToolResult
from src.agent.react_agent import ReActAgent


class MockOllamaResponse:
    """Mock response for Ollama client to simulate LLM behavior."""
    
    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0
    
    def chat(self, model, messages):
        """Return predefined responses simulating the ReAct cycle."""
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return {
                'message': {
                    'content': response
                }
            }
        else:
            # Fallback response
            return {
                'message': {
                    'content': 'Final Answer: Task completed successfully.'
                }
            }


def test_directory_tool_standalone():
    """Test the CreateDirectoryTool in isolation."""
    print("Testing CreateDirectoryTool standalone...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_path = Path(temp_dir) / "test_project" / "src" / "components"
        
        # Test tool directly
        create_tool = CreateDirectoryTool()
        result = create_tool.execute(path=str(test_path))
        
        assert result.success, f"Directory creation failed: {result.error}"
        assert test_path.exists(), "Directory was not created"
        assert test_path.is_dir(), "Path exists but is not a directory"
        
        print(f"✅ Directory created successfully: {test_path}")
        print(f"   Result: {result.content}")


def test_directory_tool_via_registry():
    """Test directory creation through the tool registry."""
    print("\nTesting directory creation via ToolRegistry...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_path = Path(temp_dir) / "registry_test" / "nested" / "structure"
        
        # Test through registry
        registry = ToolRegistry()
        action = ToolAction(
            tool_name="create_directory",
            parameters={"path": str(test_path)}
        )
        
        result = registry.execute_tool(action)
        
        assert result.success, f"Registry execution failed: {result.error}"
        assert test_path.exists(), "Directory was not created via registry"
        
        print(f"✅ Directory created via registry: {test_path}")
        print(f"   Result: {result.content}")


def test_complex_directory_structure():
    """Test creating a complex project directory structure."""
    print("\nTesting complex directory structure creation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        base_path = Path(temp_dir) / "my_project"
        
        # Define project structure
        directories = [
            "src/components",
            "src/utils",
            "src/config", 
            "tests/unit",
            "tests/integration",
            "docs/api",
            "docs/user_guide",
            "build/assets",
            "scripts/deployment"
        ]
        
        registry = ToolRegistry()
        
        # Create all directories
        for dir_path in directories:
            full_path = base_path / dir_path
            action = ToolAction(
                tool_name="create_directory",
                parameters={"path": str(full_path)}
            )
            
            result = registry.execute_tool(action)
            assert result.success, f"Failed to create {dir_path}: {result.error}"
            assert full_path.exists(), f"Directory {dir_path} was not created"
            
            print(f"   ✅ Created: {dir_path}")
        
        # Verify structure by listing the base directory
        list_action = ToolAction(
            tool_name="list_directory",
            parameters={"path": str(base_path)}
        )
        
        list_result = registry.execute_tool(list_action)
        assert list_result.success, f"Failed to list directory: {list_result.error}"
        
        print(f"\n📁 Project structure created successfully in {base_path}")
        print(f"   Base directory contents: {list_result.content}")
        
        # Verify specific directories exist
        assert (base_path / "src" / "components").exists()
        assert (base_path / "tests" / "unit").exists()
        assert (base_path / "docs" / "api").exists()
        
        print("✅ All directories verified!")


def test_agent_directory_creation_with_mock():
    """Test agent creating directories using mocked LLM responses."""
    print("\nTesting agent directory creation with mocked LLM...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_path = Path(temp_dir) / "agent_project" / "backend" / "api"
        
        # Mock LLM responses for ReAct cycle
        mock_responses = [
            # First response: Analyze and plan
            f"""Thought: I need to create a directory structure for a backend API project at {test_path}. Let me start by creating the main directory structure.
Action: create_directory
Action Input: {{"path": "{test_path}"}}""",
            
            # Second response: Create additional directories
            f"""Thought: Good, the main directory structure is created. Now let me create some additional directories for the project structure.
Action: create_directory
Action Input: {{"path": "{test_path}/models"}}""",
            
            # Third response: Create more directories
            f"""Thought: Now I'll create the routes directory.
Action: create_directory
Action Input: {{"path": "{test_path}/routes"}}""",
            
            # Final response
            f"""Thought: The directory structure has been created successfully. I have created the main backend/api directory and subdirectories for models and routes.
Final Answer: Successfully created the directory structure for the backend API project at {test_path}. The structure includes the main api directory with models and routes subdirectories."""
        ]
        
        # Create agent with mocked Ollama client
        with patch('src.agent.react_agent.OllamaClient') as mock_ollama_client:
            mock_client_instance = MagicMock()
            mock_client_instance.chat.side_effect = lambda model, messages: MockOllamaResponse(mock_responses).chat(model, messages)
            mock_ollama_client.return_value = mock_client_instance
            
            # Create agent
            agent = ReActAgent(model="test-model")
            
            # Process request
            user_input = f"Create a directory structure for a backend API project at {test_path}"
            result = agent.process_request(user_input)
            
            # Verify directories were created
            assert test_path.exists(), "Main directory was not created by agent"
            assert (test_path / "models").exists(), "Models directory was not created"
            assert (test_path / "routes").exists(), "Routes directory was not created"
            
            print(f"✅ Agent successfully created directory structure")
            print(f"   Main directory: {test_path}")
            print(f"   Subdirectories: models, routes")
            print(f"   Agent response: {result[:100]}...")


def run_all_tests():
    """Run all directory creation tests."""
    print("🧪 Running Agent Directory Creation Tests")
    print("=" * 50)
    
    try:
        test_directory_tool_standalone()
        test_directory_tool_via_registry()
        test_complex_directory_structure()
        test_agent_directory_creation_with_mock()
        
        print("\n" + "=" * 50)
        print("✅ All tests passed! Agent directory creation is working correctly.")
        print("\nTest Summary:")
        print("- ✅ Standalone directory tool")
        print("- ✅ Directory creation via registry")
        print("- ✅ Complex directory structure")
        print("- ✅ Agent with mocked LLM responses")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()