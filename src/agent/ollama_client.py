import time
from typing import Dict, List, Optional, Iterator, Any

import ollama
from ollama import ResponseError
from llm_logging import log_request, log_response, log_error


class OllamaClient:
    """Wrapper for Ollama client with enhanced functionality."""
    
    def __init__(self, host: str = "http://localhost:11434", timeout: int = 30):
        self.host = host
        self.timeout = timeout
        self.client = ollama.Client(host=host, timeout=timeout)
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models."""
        try:
            response = self.client.list()
            return response.get('models', [])
        except ResponseError as e:
            raise OllamaClientError(f"Failed to list models: {e.error}")
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model from the registry."""
        try:
            self.client.pull(model_name)
            return True
        except ResponseError as e:
            raise OllamaClientError(f"Failed to pull model {model_name}: {e.error}")
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a model."""
        try:
            self.client.delete(model_name)
            return True
        except ResponseError as e:
            raise OllamaClientError(f"Failed to delete model {model_name}: {e.error}")
    
    def model_exists(self, model_name: str) -> bool:
        """Check if a model exists locally."""
        try:
            self.client.show(model_name)
            return True
        except ResponseError:
            return False
    
    def ensure_model(self, model_name: str) -> bool:
        """Ensure a model is available, pulling if necessary."""
        if self.model_exists(model_name):
            return True
        
        try:
            self.pull_model(model_name)
            return True
        except OllamaClientError:
            return False
    
    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Chat with a model (non-streaming)."""
        if stream:
            raise ValueError("Use chat_stream for streaming responses")
        
        # Log the request
        conversation_id = log_request(model, messages, metadata={"max_retries": max_retries})
        
        try:
            response = self._chat_with_retry(model, messages, stream=False, max_retries=max_retries)
            # Log the response
            if conversation_id:
                log_response(model, response, conversation_id)
            return response
        except Exception as e:
            # Log the error
            if conversation_id:
                log_error(model, str(e), conversation_id)
            raise
    
    def chat_stream(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_retries: int = 3
    ) -> Iterator[Dict[str, Any]]:
        """Chat with a model (streaming)."""
        # Log the request
        conversation_id = log_request(model, messages, metadata={"max_retries": max_retries, "streaming": True})
        
        chunk_index = 0
        try:
            for attempt in range(max_retries):
                try:
                    # Ensure model is available
                    if not self.ensure_model(model):
                        raise OllamaClientError(f"Model {model} not available and failed to pull")
                    
                    stream = self.client.chat(
                        model=model,
                        messages=messages,
                        stream=True
                    )
                    
                    for chunk in stream:
                        # Log each streaming chunk
                        if conversation_id:
                            from llm_logging import get_logger
                            logger = get_logger()
                            if logger:
                                logger.log_streaming_chunk(model, chunk, conversation_id, chunk_index)
                            chunk_index += 1
                        yield chunk
                        
                    return  # Success, exit retry loop
                    
                except ResponseError as e:
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        error_msg = f"Chat failed after {max_retries} attempts: {e.error}"
                        if conversation_id:
                            log_error(model, error_msg, conversation_id)
                        raise OllamaClientError(error_msg)
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        error_msg = f"Unexpected error: {str(e)}"
                        if conversation_id:
                            log_error(model, error_msg, conversation_id)
                        raise OllamaClientError(error_msg)
        except Exception as e:
            # Log any unhandled errors
            if conversation_id:
                log_error(model, str(e), conversation_id)
            raise
    
    def _chat_with_retry(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Internal method to handle chat with retry logic."""
        for attempt in range(max_retries):
            try:
                # Ensure model is available
                if not self.ensure_model(model):
                    raise OllamaClientError(f"Model {model} not available and failed to pull")
                
                response = self.client.chat(
                    model=model,
                    messages=messages,
                    stream=stream
                )
                
                return response
                
            except ResponseError as e:
                if e.status_code == 404 and attempt == 0:
                    # Try to pull model on first 404
                    try:
                        self.pull_model(model)
                        continue
                    except OllamaClientError:
                        pass
                
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    raise OllamaClientError(f"Chat failed after {max_retries} attempts: {e.error}")
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise OllamaClientError(f"Unexpected error: {str(e)}")
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a model."""
        try:
            return self.client.show(model_name)
        except ResponseError as e:
            raise OllamaClientError(f"Failed to get model info for {model_name}: {e.error}")


class OllamaClientError(Exception):
    """Exception raised for Ollama client errors."""
    pass