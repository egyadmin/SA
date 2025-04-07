"""
Local AI Model Integration Module for Manus Clone

This module provides integration with local AI models including:
- Ollama models
- Local LLMs
- Hugging Face models
- Custom fine-tuned models
"""

import os
import re
import json
import time
import uuid
import logging
import tempfile
import subprocess
import requests
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LocalAIModelIntegration:
    """Local AI Model Integration class for running AI models locally"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Local AI Model Integration system
        
        Args:
            config: Configuration parameters
        """
        self.config = config
        self.models_dir = config.get('MODELS_DIR', os.path.join(os.path.expanduser('~'), '.manus_clone', 'models'))
        self.ollama_host = config.get('OLLAMA_HOST', 'http://localhost:11434')
        self.default_model = config.get('DEFAULT_MODEL', 'llama2')
        self.model_configs = config.get('MODEL_CONFIGS', {})
        self.cache_dir = config.get('CACHE_DIR', os.path.join(os.path.expanduser('~'), '.manus_clone', 'cache'))
        
        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Check for required dependencies
        self._check_dependencies()
        
        # Initialize Ollama if available
        self.ollama_available = self._check_ollama_availability()
        
        logger.info(f"Local AI Model Integration initialized with default model: {self.default_model}")
        if self.ollama_available:
            logger.info(f"Ollama is available at {self.ollama_host}")
        else:
            logger.warning("Ollama is not available. Some features may not work.")
    
    def _check_dependencies(self) -> None:
        """
        Check if required dependencies are installed
        """
        try:
            # Check Python dependencies
            required_modules = ['requests', 'transformers', 'torch', 'huggingface_hub', 'sentencepiece']
            missing_modules = []
            
            for module in required_modules:
                try:
                    __import__(module.replace('-', '_').split('.')[0])
                except ImportError:
                    missing_modules.append(module)
            
            if missing_modules:
                logger.warning(f"Missing Python modules: {', '.join(missing_modules)}. Some features may not work.")
                logger.info(f"Install missing modules with: pip install {' '.join(missing_modules)}")
            else:
                logger.info("All required Python modules are installed")
                
        except Exception as e:
            logger.error(f"Error checking dependencies: {str(e)}")
    
    def _check_ollama_availability(self) -> bool:
        """
        Check if Ollama is available
        
        Returns:
            True if Ollama is available, False otherwise
        """
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama is not available: {str(e)}")
            return False
    
    def install_ollama(self) -> bool:
        """
        Install Ollama if not already installed
        
        Returns:
            True if installation was successful, False otherwise
        """
        try:
            # Check if Ollama is already installed
            if self.ollama_available:
                logger.info("Ollama is already installed and running")
                return True
            
            # Check operating system
            import platform
            system = platform.system().lower()
            
            if system == 'linux':
                # Install Ollama on Linux
                logger.info("Installing Ollama on Linux...")
                subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
            elif system == 'darwin':
                # Install Ollama on macOS
                logger.info("Installing Ollama on macOS...")
                subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
            elif system == 'windows':
                # Install Ollama on Windows
                logger.info("Installing Ollama on Windows...")
                logger.info("Please download and install Ollama from https://ollama.com/download")
                return False
            else:
                logger.error(f"Unsupported operating system: {system}")
                return False
            
            # Check if installation was successful
            time.sleep(5)  # Wait for Ollama to start
            self.ollama_available = self._check_ollama_availability()
            
            if self.ollama_available:
                logger.info("Ollama installed successfully")
                return True
            else:
                logger.error("Failed to install Ollama")
                return False
        except Exception as e:
            logger.error(f"Error installing Ollama: {str(e)}")
            return False
    
    def start_ollama_server(self) -> bool:
        """
        Start Ollama server if not already running
        
        Returns:
            True if server was started successfully, False otherwise
        """
        try:
            # Check if Ollama is already running
            if self.ollama_available:
                logger.info("Ollama server is already running")
                return True
            
            # Start Ollama server
            logger.info("Starting Ollama server...")
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for server to start
            for _ in range(10):
                time.sleep(1)
                if self._check_ollama_availability():
                    logger.info("Ollama server started successfully")
                    self.ollama_available = True
                    return True
            
            logger.error("Failed to start Ollama server")
            return False
        except Exception as e:
            logger.error(f"Error starting Ollama server: {str(e)}")
            return False
    
    def list_available_models(self) -> Dict[str, Any]:
        """
        List available models
        
        Returns:
            Dictionary with available models information
        """
        try:
            models = {
                'ollama': [],
                'huggingface': [],
                'custom': []
            }
            
            # List Ollama models
            if self.ollama_available:
                response = requests.get(f"{self.ollama_host}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models['ollama'] = [
                        {
                            'name': model['name'],
                            'size': model.get('size', 0),
                            'modified_at': model.get('modified_at', ''),
                            'source': 'ollama'
                        }
                        for model in data.get('models', [])
                    ]
            
            # List Hugging Face models
            try:
                from huggingface_hub import list_models
                
                # List some popular models
                popular_models = [
                    'mistralai/Mistral-7B-Instruct-v0.2',
                    'meta-llama/Llama-2-7b-chat-hf',
                    'google/gemma-7b-it',
                    'microsoft/phi-2',
                    'stabilityai/stablelm-2-1_6b',
                    'bigscience/bloom-1b7',
                    'EleutherAI/pythia-1.4b',
                    'facebook/opt-1.3b'
                ]
                
                models['huggingface'] = [
                    {
                        'name': model_id,
                        'size': 'Unknown',
                        'modified_at': '',
                        'source': 'huggingface'
                    }
                    for model_id in popular_models
                ]
            except Exception as e:
                logger.warning(f"Error listing Hugging Face models: {str(e)}")
            
            # List custom models
            custom_models_dir = os.path.join(self.models_dir, 'custom')
            if os.path.exists(custom_models_dir):
                for model_dir in os.listdir(custom_models_dir):
                    model_path = os.path.join(custom_models_dir, model_dir)
                    if os.path.isdir(model_path):
                        models['custom'].append({
                            'name': model_dir,
                            'size': self._get_directory_size(model_path),
                            'modified_at': time.ctime(os.path.getmtime(model_path)),
                            'source': 'custom'
                        })
            
            return {
                'models': models,
                'total_count': sum(len(models[source]) for source in models),
                'ollama_available': self.ollama_available
            }
        except Exception as e:
            logger.error(f"Error listing available models: {str(e)}")
            return {
                'models': {'ollama': [], 'huggingface': [], 'custom': []},
                'total_count': 0,
                'ollama_available': self.ollama_available,
                'error': str(e)
            }
    
    def _get_directory_size(self, path: str) -> str:
        """
        Get the size of a directory in human-readable format
        
        Args:
            path: Directory path
            
        Returns:
            Size in human-readable format
        """
        total_size = 0
        for dirpath, _, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        
        # Convert to human-readable format
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if total_size < 1024:
                return f"{total_size:.2f} {unit}"
            total_size /= 1024
        
        return f"{total_size:.2f} PB"
    
    def pull_ollama_model(self, model_name: str) -> Dict[str, Any]:
        """
        Pull an Ollama model
        
        Args:
            model_name: Name of the model to pull
            
        Returns:
            Dictionary with pull status
        """
        try:
            if not self.ollama_available:
                if not self.start_ollama_server():
                    return {
                        'success': False,
                        'message': 'Ollama server is not available',
                        'model': model_name
                    }
            
            logger.info(f"Pulling Ollama model: {model_name}")
            
            # Pull model
            response = requests.post(
                f"{self.ollama_host}/api/pull",
                json={'name': model_name},
                stream=True
            )
            
            if response.status_code != 200:
                return {
                    'success': False,
                    'message': f"Failed to pull model: {response.text}",
                    'model': model_name
                }
            
            # Process streaming response
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if 'error' in data:
                        return {
                            'success': False,
                            'message': data['error'],
                            'model': model_name
                        }
                    
                    # Log progress
                    if 'status' in data:
                        logger.info(f"Pull status: {data['status']}")
                    
                    # Check if completed
                    if data.get('status') == 'success':
                        return {
                            'success': True,
                            'message': 'Model pulled successfully',
                            'model': model_name
                        }
            
            return {
                'success': True,
                'message': 'Model pulled successfully',
                'model': model_name
            }
        except Exception as e:
            logger.error(f"Error pulling Ollama model: {str(e)}")
            return {
                'success': False,
                'message': str(e),
                'model': model_name
            }
    
    def download_huggingface_model(self, model_id: str) -> Dict[str, Any]:
        """
        Download a Hugging Face model
        
        Args:
            model_id: Hugging Face model ID
            
        Returns:
            Dictionary with download status
        """
        try:
            from huggingface_hub import snapshot_download
            
            logger.info(f"Downloading Hugging Face model: {model_id}")
            
            # Create directory for the model
            model_dir = os.path.join(self.models_dir, 'huggingface', model_id.replace('/', '_'))
            os.makedirs(os.path.dirname(model_dir), exist_ok=True)
            
            # Download model
            local_dir = snapshot_download(
                repo_id=model_id,
                local_dir=model_dir,
                local_dir_use_symlinks=False
            )
            
            return {
                'success': True,
                'message': 'Model downloaded successfully',
                'model': model_id,
                'local_dir': local_dir
            }
        except Exception as e:
            logger.error(f"Error downloading Hugging Face model: {str(e)}")
            return {
                'success': False,
                'message': str(e),
                'model': model_id
            }
    
    def generate_text_with_ollama(self, prompt: str, model: Optional[str] = None, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate text using Ollama
        
        Args:
            prompt: Input prompt
            model: Model name (if None, uses default model)
            options: Generation options
            
        Returns:
            Dictionary with generated text
        """
        try:
            if not self.ollama_available:
                if not self.start_ollama_server():
                    return {
                        'success': False,
                        'message': 'Ollama server is not available',
                        'prompt': prompt
                    }
            
            # Use default model if not specified
            model = model or self.default_model
            
            # Default options
            default_options = {
                'temperature': 0.7,
                'top_p': 0.9,
                'top_k': 40,
                'max_tokens': 500,
                'stop': [],
                'stream': False
            }
            
            # Merge with provided options
            if options:
                default_options.update(options)
            
            # Prepare request
            request_data = {
                'model': model,
                'prompt': prompt,
                'options': default_options,
                'stream': default_options.pop('stream', False)
            }
            
            # Gen
(Content truncated due to size limit. Use line ranges to read in chunks)