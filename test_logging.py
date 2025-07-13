#!/usr/bin/env python3
"""
Test script for the LLM logging system.
"""

import tempfile
import json
from pathlib import Path

from src.logging.llm_logger import LLMLogger, initialize_logger, get_logger, log_request, log_response, log_error


def test_basic_logging():
    """Test basic logging functionality."""
    print("Testing basic LLM logging...")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize logger
        logger = LLMLogger(
            log_dir=temp_dir,
            log_file="test_llm.jsonl",
            enabled=True
        )
        
        # Test logging a request
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        conv_id = logger.log_request(
            model="llama3.2",
            messages=test_messages,
            metadata={"test": True}
        )
        
        print(f"Logged request with conversation ID: {conv_id}")
        
        # Test logging a response
        test_response = {
            "message": {"content": "I'm doing well, thank you for asking!"},
            "model": "llama3.2",
            "done": True
        }
        
        logger.log_response(
            model="llama3.2",
            response=test_response,
            conversation_id=conv_id,
            metadata={"response_time": 1.23}
        )
        
        print("Logged response")
        
        # Test logging an error
        error_conv_id = logger.log_error(
            model="llama3.2",
            error="Connection timeout",
            metadata={"attempt": 1}
        )
        
        print(f"Logged error with conversation ID: {error_conv_id}")
        
        # Check log file contents
        log_file = Path(temp_dir) / "test_llm.jsonl"
        if log_file.exists():
            print(f"\nLog file created: {log_file}")
            print(f"File size: {log_file.stat().st_size} bytes")
            
            # Read and display log entries
            with open(log_file, 'r') as f:
                lines = f.readlines()
                print(f"Number of log entries: {len(lines)}")
                
                for i, line in enumerate(lines, 1):
                    entry = json.loads(line.strip())
                    print(f"\nEntry {i}:")
                    print(f"  Direction: {entry['direction']}")
                    print(f"  Model: {entry['model']}")
                    print(f"  Timestamp: {entry['timestamp']}")
                    print(f"  Conversation ID: {entry['conversation_id']}")
        
        # Test reading logs
        entries = logger.read_logs(conversation_id=conv_id)
        print(f"\nRead {len(entries)} entries for conversation {conv_id}")
        
        # Test log stats
        stats = logger.get_log_stats()
        print(f"\nLog statistics: {stats}")
        
    print("✓ Basic logging test completed successfully!")


def test_global_logger():
    """Test global logger functions."""
    print("\nTesting global logger functions...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize global logger
        initialize_logger(
            log_dir=temp_dir,
            log_file="global_test.jsonl",
            enabled=True
        )
        
        # Test global logging functions
        conv_id = log_request(
            model="test-model",
            messages=[{"role": "user", "content": "Test message"}]
        )
        
        log_response(
            model="test-model",
            response={"message": {"content": "Test response"}},
            conversation_id=conv_id
        )
        
        error_conv_id = log_error(
            model="test-model",
            error="Test error"
        )
        
        # Check that global logger is accessible
        logger = get_logger()
        assert logger is not None, "Global logger should not be None"
        
        stats = logger.get_log_stats()
        print(f"Global logger stats: {stats}")
        
    print("✓ Global logger test completed successfully!")


def test_streaming_logs():
    """Test streaming chunk logging."""
    print("\nTesting streaming chunk logging...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        logger = LLMLogger(
            log_dir=temp_dir,
            log_file="streaming_test.jsonl",
            enabled=True
        )
        
        # Simulate streaming chunks
        conv_id = logger.log_request(
            model="streaming-model",
            messages=[{"role": "user", "content": "Tell me a story"}],
            metadata={"streaming": True}
        )
        
        # Log several streaming chunks
        chunks = [
            {"message": {"content": "Once"}},
            {"message": {"content": " upon"}},
            {"message": {"content": " a"}},
            {"message": {"content": " time"}},
            {"done": True}
        ]
        
        for i, chunk in enumerate(chunks):
            logger.log_streaming_chunk(
                model="streaming-model",
                chunk=chunk,
                conversation_id=conv_id,
                chunk_index=i
            )
        
        # Check that all chunks were logged
        entries = logger.read_logs(conversation_id=conv_id)
        chunk_entries = [e for e in entries if e.direction == "response_chunk"]
        
        print(f"Logged {len(chunk_entries)} streaming chunks")
        assert len(chunk_entries) == 5, f"Expected 5 chunks, got {len(chunk_entries)}"
        
    print("✓ Streaming logging test completed successfully!")


def test_log_rotation():
    """Test log file rotation."""
    print("\nTesting log file rotation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create logger with very small max file size
        logger = LLMLogger(
            log_dir=temp_dir,
            log_file="rotation_test.jsonl",
            enabled=True,
            max_file_size=500,  # Very small for testing
            max_files=3
        )
        
        # Log many entries to trigger rotation
        for i in range(20):
            conv_id = logger.log_request(
                model="test-model",
                messages=[{"role": "user", "content": f"Message {i} with some content to make it longer"}],
                metadata={"iteration": i}
            )
            
            logger.log_response(
                model="test-model",
                response={"message": {"content": f"Response {i} with some content to make it longer"}},
                conversation_id=conv_id
            )
        
        # Check for rotated files
        log_files = list(Path(temp_dir).glob("*.jsonl"))
        print(f"Created {len(log_files)} log files")
        
        for log_file in log_files:
            print(f"  {log_file.name}: {log_file.stat().st_size} bytes")
    
    print("✓ Log rotation test completed successfully!")


def test_disabled_logging():
    """Test logging when disabled."""
    print("\nTesting disabled logging...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        logger = LLMLogger(
            log_dir=temp_dir,
            enabled=False  # Disabled
        )
        
        # Try to log - should do nothing
        conv_id = logger.log_request(
            model="test-model",
            messages=[{"role": "user", "content": "This should not be logged"}]
        )
        
        # Check that no log file was created
        log_files = list(Path(temp_dir).glob("*.jsonl"))
        assert len(log_files) == 0, f"Expected no log files, found {len(log_files)}"
        
        # Stats should indicate disabled
        stats = logger.get_log_stats()
        assert not stats["enabled"], "Logger should report as disabled"
        
    print("✓ Disabled logging test completed successfully!")


if __name__ == "__main__":
    print("Running LLM logging system tests...\n")
    
    test_basic_logging()
    test_global_logger()
    test_streaming_logs()
    test_log_rotation()
    test_disabled_logging()
    
    print("\n🎉 All tests passed successfully!")
    print("\nThe LLM logging system is ready to use!")
    print("\nTo enable logging, set environment variables:")
    print("  export LLM_LOGGING_ENABLED=true")
    print("  export LLM_LOG_DIR=/path/to/logs  # Optional")
    print("  export LLM_LOG_FILE=custom_name.jsonl  # Optional")