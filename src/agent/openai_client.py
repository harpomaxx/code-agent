import time
from typing import Dict, List, Optional, Iterator, Any

from openai import OpenAI
from llm_logging import log_request, log_response, log_error


class OpenAIClient:
    """Wrapper for OpenAI client with enhanced functionality."""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, timeout: int = 30):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout
        )
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models."""
        try:
            response = self.client.models.list()
            return [{"id": model.id, "name": model.id, "created": model.created} for model in response.data]
        except Exception as e:
            raise OpenAIClientError(f"Failed to list models: {str(e)}")
    
    def model_exists(self, model_name: str) -> bool:
        """Check if a model exists."""
        try:
            self.client.models.retrieve(model_name)
            return True
        except Exception:
            return False
    
    def ensure_model(self, model_name: str) -> bool:
        """Ensure a model is available. For OpenAI, we just check if it exists."""
        return self.model_exists(model_name)
    
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
                    stream = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        stream=True
                    )
                    
                    for chunk in stream:
                        # Convert OpenAI chunk to ollama-like format
                        chunk_dict = {
                            "message": {
                                "role": "assistant",
                                "content": chunk.choices[0].delta.content or ""
                            },
                            "done": chunk.choices[0].finish_reason is not None
                        }
                        
                        # Log each streaming chunk
                        if conversation_id:
                            from llm_logging import get_logger
                            logger = get_logger()
                            if logger:
                                logger.log_streaming_chunk(model, chunk_dict, conversation_id, chunk_index)
                            chunk_index += 1
                        
                        yield chunk_dict
                        
                    return  # Success, exit retry loop
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        error_msg = f"Chat failed after {max_retries} attempts: {str(e)}"
                        if conversation_id:
                            log_error(model, error_msg, conversation_id)
                        raise OpenAIClientError(error_msg)
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
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=stream
                )
                
                # Convert OpenAI response to ollama-like format
                if hasattr(response, 'choices') and response.choices:
                    return {
                        "message": {
                            "role": "assistant",
                            "content": response.choices[0].message.content
                        },
                        "done": True
                    }
                else:
                    raise OpenAIClientError("No response received")
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    raise OpenAIClientError(f"Chat failed after {max_retries} attempts: {str(e)}")
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a model."""
        try:
            model = self.client.models.retrieve(model_name)
            return {
                "id": model.id,
                "created": model.created,
                "owned_by": model.owned_by
            }
        except Exception as e:
            raise OpenAIClientError(f"Failed to get model info for {model_name}: {str(e)}")


class OpenAIClientError(Exception):
    """Exception raised for OpenAI client errors."""
    pass