"""
AI Models Integration Module for Manus Clone

This module handles integration with external AI models including:
- OpenAI models (GPT-4, GPT-3.5, etc.)
- Anthropic Claude models
- Local AI models via Ollama
- Configuration and management of API keys
- Model selection and fallback mechanisms
- Response streaming
"""

import os
import json
import time
import logging
import asyncio
import aiohttp
import requests
from typing import Dict, List, Any, Optional, Union, Callable, AsyncGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AIModelsIntegration:
    """AI Models Integration class for external AI models"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the AI Models Integration with configuration
        
        Args:
            config: Dictionary containing configuration parameters including API keys
        """
        self.config = config
        self.openai_api_key = config.get('OPENAI_API_KEY', os.environ.get('OPENAI_API_KEY', ''))
        self.anthropic_api_key = config.get('ANTHROPIC_API_KEY', os.environ.get('ANTHROPIC_API_KEY', ''))
        self.local_model_url = config.get('LOCAL_MODEL_URL', 'http://localhost:11434')
        
        # Default model configurations
        self.default_openai_model = config.get('DEFAULT_OPENAI_MODEL', 'gpt-4o')
        self.default_claude_model = config.get('DEFAULT_CLAUDE_MODEL', 'claude-3-opus-20240229')
        self.default_local_model = config.get('DEFAULT_LOCAL_MODEL', 'llama3')
        
        # Validate API keys and connections
        self.openai_available = self._check_openai_connection()
        self.anthropic_available = self._check_anthropic_connection()
        self.local_model_available = self._check_local_model_connection()
        
        available_models = []
        if self.openai_available:
            available_models.append("OpenAI")
        if self.anthropic_available:
            available_models.append("Anthropic Claude")
        if self.local_model_available:
            available_models.append("Local Model")
            
        if not available_models:
            logger.warning("No AI models available. AI model integration will be limited.")
        else:
            logger.info(f"AI Models Integration initialized with {', '.join(available_models)}")
    
    def _check_openai_connection(self) -> bool:
        """Check if OpenAI API is accessible with the provided key"""
        if not self.openai_api_key:
            return False
            
        try:
            import openai
            client = openai.OpenAI(api_key=self.openai_api_key)
            # Just check if we can list models (lightweight request)
            client.models.list(limit=1)
            return True
        except Exception as e:
            logger.warning(f"OpenAI connection failed: {str(e)}")
            return False
    
    def _check_anthropic_connection(self) -> bool:
        """Check if Anthropic API is accessible with the provided key"""
        if not self.anthropic_api_key:
            return False
            
        try:
            headers = {
                "x-api-key": self.anthropic_api_key,
                "anthropic-version": "2023-06-01"
            }
            response = requests.get("https://api.anthropic.com/v1/models", headers=headers)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Anthropic connection failed: {str(e)}")
            return False
    
    def _check_local_model_connection(self) -> bool:
        """Check if local model server (Ollama) is accessible"""
        try:
            response = requests.get(f"{self.local_model_url}/api/tags")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Local model connection failed: {str(e)}")
            return False
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """
        Get a list of all available AI models
        
        Returns:
            Dictionary of available models grouped by provider
        """
        available_models = {}
        
        if self.openai_available:
            try:
                import openai
                client = openai.OpenAI(api_key=self.openai_api_key)
                models_list = client.models.list()
                openai_models = [model.id for model in models_list.data if model.id.startswith("gpt")]
                available_models["openai"] = openai_models
            except Exception as e:
                logger.error(f"Error fetching OpenAI models: {str(e)}")
                available_models["openai"] = [
                    "gpt-4o",
                    "gpt-4-turbo",
                    "gpt-4",
                    "gpt-3.5-turbo"
                ]
        
        if self.anthropic_available:
            try:
                headers = {
                    "x-api-key": self.anthropic_api_key,
                    "anthropic-version": "2023-06-01"
                }
                response = requests.get("https://api.anthropic.com/v1/models", headers=headers)
                if response.status_code == 200:
                    claude_models = [model["id"] for model in response.json()["models"]]
                    available_models["claude"] = claude_models
                else:
                    raise Exception(f"API returned status code {response.status_code}")
            except Exception as e:
                logger.error(f"Error fetching Claude models: {str(e)}")
                available_models["claude"] = [
                    "claude-3-opus-20240229",
                    "claude-3-sonnet-20240229",
                    "claude-3-haiku-20240307"
                ]
        
        if self.local_model_available:
            try:
                response = requests.get(f"{self.local_model_url}/api/tags")
                if response.status_code == 200:
                    local_models = [model["name"] for model in response.json()["models"]]
                    available_models["local"] = local_models
                else:
                    raise Exception(f"API returned status code {response.status_code}")
            except Exception as e:
                logger.error(f"Error fetching local models: {str(e)}")
                available_models["local"] = ["llama3", "mistral", "gemma"]
        
        return available_models
    
    async def generate_text(self, 
                           prompt: str, 
                           model: Optional[str] = None,
                           provider: Optional[str] = None,
                           system_message: Optional[str] = None,
                           temperature: float = 0.7,
                           max_tokens: Optional[int] = None,
                           stream: bool = False,
                           callback: Optional[Callable[[str], None]] = None) -> Union[str, AsyncGenerator[str, None]]:
        """
        Generate text using the specified AI model
        
        Args:
            prompt: User prompt or question
            model: Specific model to use (if None, uses default model for the provider)
            provider: AI provider to use ('openai', 'claude', or 'local', if None tries available providers)
            system_message: System message or instructions for the AI
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            callback: Callback function for streaming responses
            
        Returns:
            Generated text or async generator for streaming
        """
        # Determine provider if not specified
        if not provider:
            if self.openai_available:
                provider = "openai"
            elif self.anthropic_available:
                provider = "claude"
            elif self.local_model_available:
                provider = "local"
            else:
                raise ValueError("No AI providers available. Please provide API keys or connect to a local model.")
        
        # Validate provider
        if provider.lower() == "openai" and not self.openai_available:
            raise ValueError("OpenAI API key not provided or connection failed. Cannot use OpenAI models.")
        elif provider.lower() == "claude" and not self.anthropic_available:
            raise ValueError("Anthropic API key not provided or connection failed. Cannot use Claude models.")
        elif provider.lower() == "local" and not self.local_model_available:
            raise ValueError("Local model connection failed. Cannot use local models.")
        
        # Determine model if not specified
        if not model:
            if provider.lower() == "openai":
                model = self.default_openai_model
            elif provider.lower() == "claude":
                model = self.default_claude_model
            elif provider.lower() == "local":
                model = self.default_local_model
        
        # Generate text based on provider
        if provider.lower() == "openai":
            return await self._generate_openai(prompt, model, system_message, temperature, max_tokens, stream, callback)
        elif provider.lower() == "claude":
            return await self._generate_claude(prompt, model, system_message, temperature, max_tokens, stream, callback)
        elif provider.lower() == "local":
            return await self._generate_local(prompt, model, system_message, temperature, max_tokens, stream, callback)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    async def _generate_openai(self,
                              prompt: str,
                              model: str,
                              system_message: Optional[str],
                              temperature: float,
                              max_tokens: Optional[int],
                              stream: bool,
                              callback: Optional[Callable[[str], None]]) -> Union[str, AsyncGenerator[str, None]]:
        """
        Generate text using OpenAI models
        
        Args:
            prompt: User prompt or question
            model: OpenAI model to use
            system_message: System message or instructions
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            callback: Callback function for streaming responses
            
        Returns:
            Generated text or async generator for streaming
        """
        try:
            import openai
            
            # Configure OpenAI client
            client = openai.AsyncOpenAI(api_key=self.openai_api_key)
            
            # Prepare messages
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            # Prepare request parameters
            params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            
            if max_tokens:
                params["max_tokens"] = max_tokens
            
            # Handle streaming
            if stream:
                async def response_generator():
                    try:
                        response_stream = await client.chat.completions.create(
                            **params,
                            stream=True
                        )
                        
                        full_response = ""
                        async for chunk in response_stream:
                            if chunk.choices and chunk.choices[0].delta.content:
                                content = chunk.choices[0].delta.content
                                full_response += content
                                
                                if callback:
                                    callback(content)
                                
                                yield content
                    except Exception as e:
                        logger.error(f"Error in OpenAI streaming: {str(e)}")
                        raise
                
                return response_generator()
            else:
                # Non-streaming response
                response = await client.chat.completions.create(**params)
                return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in OpenAI completion: {str(e)}")
            error_msg = f"OpenAI API error: {str(e)}"
            if "API key" in str(e).lower():
                error_msg = "Invalid OpenAI API key. Please check your API key and try again."
            elif "rate limit" in str(e).lower():
                error_msg = "OpenAI rate limit exceeded. Please try again later."
            raise ValueError(error_msg)
    
    async def _generate_claude(self,
                              prompt: str,
                              model: str,
                              system_message: Optional[str],
                              temperature: float,
                              max_tokens: Optional[int],
                              stream: bool,
                              callback: Optional[Callable[[str], None]]) -> Union[str, AsyncGenerator[str, None]]:
        """
        Generate text using Anthropic Claude models
        
        Args:
            prompt: User prompt or question
            model: Claude model to use
            system_message: System message or instructions
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            callback: Callback function for streaming responses
            
        Returns:
            Generated text or async generator for streaming
        """
        try:
            # Prepare API endpoint
            api_url = "https://api.anthropic.com/v1/messages"
            
            # Prepare headers
            headers = {
                "x-api-key": self.anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            
            # Prepare request body
            body = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature
            }
            
            if system_message:
                body["system"] = system_message
            
            if max_tokens:
                body["max_tokens"] = max_tokens
            
            # Handle streaming
            if stream:
                body["stream"] = True
                
                async def response_generator():
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.post(api_url, headers=headers, json=body) as response:
                                if response.status != 200:
                                    error_text = await response.text()
                                    raise ValueError(f"Claude API error: {response.status} - {error_text}")
                                
                                # Process the streaming response
                                full_response = ""
                                async for line in response.content:
                                    line = line.decode('utf-8').strip()
                                    if not line or line == "data: [DONE]":
                                        continue
                                    
                                    if line.startswith("data: "):
                                        data = json.loads(line[6:])
                                        if data["type"] == "content_block_delta":
                                            content = data["delta"]["text"]
                                            full_response += content
                                            
                                            if callback:
                                                callback(content)
                                            
                                            yield content
                    except Exception as e:
                        logger.error(f"Error in Claude streaming: {str(e)}")
                        raise
                
                return response_generator()
            else:
                # Non-streaming response
                response = requests.post(api_url, headers=headers, json=body)
                response.raise_for_status()
                return response.json()["content"][0]["text"]
        except Exception as e:
            logger.error(f"Error in Claude completion: {str(e)}")
            error_msg = f"Claude API error: {str(e)}"
            if "api key" in str(e).lower():
                error_msg = "Invalid Anthropic API key. Please check your API key and try again."
            elif "rate limit" in str(e).lower():
                error_msg = "Anthropic rate limit exceeded. Please try again later."
            raise ValueError(error_msg)
    
    async def _generate_local(self,
                             prompt: str,
                             model: str,
                             system_message: Optional[str],
                             temperature: float,
                             max_tokens: Optional[int],
                             stream: bool,
                             callback: Optional[Callable[[str], None]]) -> Union[str, AsyncGenerator[str, None]]:
        """
        Generate text using local models via Ollama
        
        Args:
            prompt: User prompt or question
            model: Local model to use
            system_message: System message or instructions
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            callback: Callback function for streaming responses
            
        Returns:
            Generated text or async generator for streaming
        """
        try:
            # Prepare API endpoint
            api_url = f"{self.local_model_url}/api/generate"
            
            # Prepare request body
            body = {
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "stream": stream
            }
            
            if system_message:
                body["system"] = system_message
            
            if max_tokens:
                body["max_tokens"] = max_tokens
            
            # Handle streaming
            if stream:
                async def response_generator():
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.post(api_url, json=body) as response:
                                if response.status != 200:
                                    error_text = await response.text()
                                    raise ValueError(f"Local model API error: {response.status} - {error_text}")
                                
                                # Process the streaming response
                                full_response = ""
                                async for line in response.content:
                                    line = line.decode('utf-8').strip()
                                    if not line:
                                        continue
                                    
                                    data = json.loads(line)
                                    if "response" in data:
                                        content = data["response"]
                                        full_response += content
                                        
                                        if callback:
                                            callback(content)
                                        
                                        yield content
                    except Exception as e:
                        logger.error(f"Error in local model streaming: {str(e)}")
                        raise
                
                return response_generator()
            else:
                # Non-streaming response
                response = requests.post(api_url, json=body)
                response.raise_for_status()
                return response.json()["response"]
        except Exception as e:
            logger.error(f"Error in local model completion: {str(e)}")
            error_msg = f"Local model API error: {str(e)}"
            if "connection" in str(e).lower():
                error_msg = "Cannot connect to local model server. Please make sure Ollama is running."
            raise ValueError(error_msg)
    
    def set_api_key(self, provider: str, api_key: str) -> bool:
        """
        Set or update API key for a provider
        
        Args:
            provider: Provider name ('openai' or 'claude')
            api_key: API key to set
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if provider.lower() == "openai":
                self.openai_api_key = api_key
                self.openai_available = self._check_openai_connection()
                return self.openai_available
            elif provider.lower() in ["claude", "anthropic"]:
                self.anthropic_api_key = api_key
                self.anthropic_available = self._check_anthropic_connection()
                return self.anthropic_available
            else:
                logger.error(f"Unsupported provider: {provider}")
                return False
        except Exception as e:
            logger.error(f"Error setting API key: {str(e)}")
            return False
    
    def set_local_model_url(self, url: str) -> bool:
        """
        Set or update the URL for the local model server
        
        Args:
            url: URL of the local model server
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.local_model_url = url
            self.local_model_available = self._check_local_model_connection()
            return self.local_model_available
        except Exception as e:
            logger.error(f"Error setting local model URL: {str(e)}")
            return False
    
    def validate_api_key(self, provider: str, api_key: str) -> bool:
        """
        Validate an API key for a provider
        
        Args:
            provider: Provider name ('openai' or 'claude')
            api_key: API key to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if provider.lower() == "openai":
                # Test OpenAI API key with a simple request
                import openai
                client = openai.OpenAI(api_key=api_key)
                models = client.models.list(limit=1)
                return True
            elif provider.lower() in ["claude", "anthropic"]:
                # Test Claude API key with a simple request
                headers = {
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01"
                }
                response = requests.get("https://api.anthropic.com/v1/models", headers=headers)
                return response.status_code == 200
            else:
                logger.error(f"Unsupported provider: {provider}")
                return False
        except Exception as e:
            logger.error(f"Error validating API key: {str(e)}")
            return False
    
    async def compare_responses(self, 
                               prompt: str, 
                               models: List[Dict[str, str]],
                               system_message: Optional[str] = None) -> Dict[str, str]:
        """
        Compare responses from multiple models for the same prompt
        
        Args:
            prompt: User prompt or question
            models: List of dictionaries with 'provider' and 'model' keys
            system_message: System message or instructions
            
        Returns:
            Dictionary mapping model identifiers to responses
        """
        results = {}
        tasks = []
        
        for model_info in models:
            provider = model_info["provider"]
            model = model_info["model"]
            model_id = f"{provider}/{model}"
            
            task = asyncio.create_task(
                self.generate_text(
                    prompt=prompt,
                    model=model,
                    provider=provider,
                    system_message=system_message,
                    stream=False
                )
            )
            tasks.append((model_id, task))
        
        for model_id, task in tasks:
            try:
                response = await task
                results[model_id] = response
            except Exception as e:
                results[model_id] = f"Error: {str(e)}"
        
        return results
    
    def get_model_status(self) -> Dict[str, bool]:
        """
        Get the status of all AI model providers
        
        Returns:
            Dictionary mapping provider names to availability status
        """
        return {
            "openai": self.openai_available,
            "claude": self.anthropic_available,
            "local": self.local_model_available
        }
