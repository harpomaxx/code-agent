#!/usr/bin/env python3
"""
Test script to verify OpenAI migration works correctly.
"""
import os
import sys
sys.path.insert(0, 'src')

from agent.react_agent import ReActAgent
from agent.openai_client import OpenAIClient
from config.settings import config

def test_openai_client():
    """Test OpenAI client basic functionality."""
    print("Testing OpenAI client...")
    
    # Check if API key is available
    api_key = os.getenv('LLM_API_KEY') or os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ No API key found. Set LLM_API_KEY or OPENAI_API_KEY environment variable.")
        return False
    
    try:
        client = OpenAIClient(api_key=api_key)
        print("✅ OpenAI client initialized successfully")
        return True
    except Exception as e:
        print(f"❌ OpenAI client initialization failed: {e}")
        return False

def test_react_agent():
    """Test ReAct agent initialization."""
    print("\nTesting ReAct agent...")
    
    # Check if API key is available
    api_key = os.getenv('LLM_API_KEY') or os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ No API key found. Set LLM_API_KEY or OPENAI_API_KEY environment variable.")
        return False
    
    try:
        agent = ReActAgent(api_key=api_key)
        print("✅ ReAct agent initialized successfully")
        return True
    except Exception as e:
        print(f"❌ ReAct agent initialization failed: {e}")
        return False

def test_config():
    """Test configuration system."""
    print("\nTesting configuration...")
    
    try:
        print(f"Default model: {config.llm.default_model}")
        print(f"Provider: {config.llm.provider}")
        print(f"Timeout: {config.llm.timeout}")
        print("✅ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing OpenAI migration...\n")
    
    tests = [
        test_config,
        test_openai_client,
        test_react_agent,
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! OpenAI migration is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())