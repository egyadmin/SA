"""
Generative AI Module for Manus Clone

This module implements generative AI capabilities including:
- Image generation
- Creative content creation
- Advanced text generation
- Output quality enhancement
"""

import os
import json
import time
import logging
import uuid
import base64
import io
import requests
from typing import Dict, List, Any, Optional, Union, Tuple
from PIL import Image
from flask import Flask, request, jsonify, send_file
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GenerativeAIManager:
    """Manages generative AI capabilities for the Manus Clone application"""
    
    def __init__(self, app: Flask, config: Dict[str, Any]):
        """
        Initialize the Generative AI Manager with configuration
        
        Args:
            app: Flask application instance
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.app = app
        
        # Initialize storage directories
        self.gen_ai_dir = os.path.join(config.get('data_dir', 'data'), 'generative_ai')
        os.makedirs(self.gen_ai_dir, exist_ok=True)
        
        # Image generation storage
        self.images_dir = os.path.join(self.gen_ai_dir, 'images')
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Text generation storage
        self.texts_dir = os.path.join(self.gen_ai_dir, 'texts')
        os.makedirs(self.texts_dir, exist_ok=True)
        
        # Creative content storage
        self.creative_dir = os.path.join(self.gen_ai_dir, 'creative')
        os.makedirs(self.creative_dir, exist_ok=True)
        
        # Initialize models
        self._init_models()
        
        # Register routes
        self._register_routes()
        
        logger.info("Generative AI Manager initialized successfully")
    
    def _init_models(self):
        """Initialize AI models based on configuration"""
        # Check for available GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize image generation models
        self.image_models = {}
        self._init_image_models()
        
        # Initialize text generation models
        self.text_models = {}
        self._init_text_models()
        
        # Initialize creative content models
        self.creative_models = {}
        self._init_creative_models()
    
    def _init_image_models(self):
        """Initialize image generation models"""
        # Check if Stable Diffusion should be used
        if self.config.get('use_stable_diffusion', True):
            try:
                from diffusers import StableDiffusionPipeline
                
                # Initialize Stable Diffusion model
                model_id = self.config.get('stable_diffusion_model', "runwayml/stable-diffusion-v1-5")
                
                # Load the model (with low memory usage for CPU)
                if self.device == "cpu":
                    self.image_models['stable_diffusion'] = StableDiffusionPipeline.from_pretrained(
                        model_id,
                        torch_dtype=torch.float32
                    )
                else:
                    self.image_models['stable_diffusion'] = StableDiffusionPipeline.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16
                    ).to(self.device)
                
                logger.info(f"Stable Diffusion model initialized: {model_id}")
            except Exception as e:
                logger.error(f"Error initializing Stable Diffusion: {str(e)}")
                # Fallback to API-based image generation
                self.image_models['stable_diffusion'] = None
        
        # Check if DALL-E mini should be used (via Hugging Face)
        if self.config.get('use_dalle_mini', True):
            try:
                from transformers import pipeline
                
                # Initialize DALL-E mini model
                model_id = self.config.get('dalle_mini_model', "dalle-mini/dalle-mini")
                
                # This is a placeholder - actual implementation would use the appropriate model
                # DALL-E mini is quite large and may not be practical to run locally
                # In a real implementation, you might use the Hugging Face API instead
                
                logger.info(f"DALL-E mini model initialized: {model_id}")
            except Exception as e:
                logger.error(f"Error initializing DALL-E mini: {str(e)}")
                # Fallback to API-based image generation
        
        # Initialize OpenAI DALL-E integration if configured
        if self.config.get('use_openai_dalle', False) and self.config.get('openai_api_key'):
            self.image_models['dalle'] = {
                'api_key': self.config.get('openai_api_key'),
                'model': self.config.get('dalle_model', 'dall-e-3')
            }
            logger.info(f"OpenAI DALL-E integration initialized: {self.image_models['dalle']['model']}")
    
    def _init_text_models(self):
        """Initialize text generation models"""
        # Check if local text generation models should be used
        if self.config.get('use_local_text_models', True):
            try:
                from transformers import pipeline
                
                # Initialize text generation pipeline with a smaller model suitable for local use
                model_id = self.config.get('local_text_model', "gpt2")
                self.text_models['local'] = pipeline('text-generation', model=model_id, device=0 if self.device == "cuda" else -1)
                
                logger.info(f"Local text generation model initialized: {model_id}")
            except Exception as e:
                logger.error(f"Error initializing local text model: {str(e)}")
                self.text_models['local'] = None
        
        # Initialize OpenAI integration if configured
        if self.config.get('use_openai', True) and self.config.get('openai_api_key'):
            import openai
            openai.api_key = self.config.get('openai_api_key')
            self.text_models['openai'] = {
                'api_key': self.config.get('openai_api_key'),
                'model': self.config.get('openai_model', 'gpt-4o')
            }
            logger.info(f"OpenAI integration initialized: {self.text_models['openai']['model']}")
        
        # Initialize Anthropic integration if configured
        if self.config.get('use_anthropic', False) and self.config.get('anthropic_api_key'):
            from anthropic import Anthropic
            self.text_models['anthropic'] = {
                'client': Anthropic(api_key=self.config.get('anthropic_api_key')),
                'model': self.config.get('anthropic_model', 'claude-3-opus-20240229')
            }
            logger.info(f"Anthropic integration initialized: {self.text_models['anthropic']['model']}")
    
    def _init_creative_models(self):
        """Initialize creative content generation models"""
        # Check if music generation should be used
        if self.config.get('use_music_generation', False):
            try:
                # Placeholder for music generation model
                # In a real implementation, you might use a model like MusicGen or Jukebox
                logger.info("Music generation initialized")
            except Exception as e:
                logger.error(f"Error initializing music generation: {str(e)}")
        
        # Check if code generation should be used
        if self.config.get('use_code_generation', True):
            try:
                from transformers import pipeline
                
                # Initialize code generation pipeline
                model_id = self.config.get('code_model', "Salesforce/codegen-350M-mono")
                self.creative_models['code'] = pipeline('text-generation', model=model_id, device=0 if self.device == "cuda" else -1)
                
                logger.info(f"Code generation model initialized: {model_id}")
            except Exception as e:
                logger.error(f"Error initializing code generation model: {str(e)}")
                self.creative_models['code'] = None
    
    def _register_routes(self):
        """Register HTTP routes for generative AI features"""
        
        @self.app.route('/api/generative/image', methods=['POST'])
        def generate_image():
            """Generate an image from a text prompt"""
            data = request.get_json()
            
            if 'prompt' not in data:
                return jsonify({'success': False, 'error': 'Prompt is required'}), 400
            
            prompt = data['prompt']
            model = data.get('model', 'stable_diffusion')
            size = data.get('size', 512)  # Default size 512x512
            num_images = min(data.get('num_images', 1), 4)  # Limit to 4 images max
            
            try:
                # Generate images
                image_paths = self._generate_images(prompt, model, size, num_images)
                
                # Create response with image URLs
                image_urls = [f"/api/generative/image/{os.path.basename(path)}" for path in image_paths]
                
                return jsonify({
                    'success': True,
                    'images': image_urls,
                    'prompt': prompt,
                    'model': model
                })
            
            except Exception as e:
                logger.error(f"Error generating images: {str(e)}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/generative/image/<image_id>', methods=['GET'])
        def get_generated_image(image_id):
            """Get a generated image by ID"""
            image_path = os.path.join(self.images_dir, image_id)
            
            if not os.path.exists(image_path):
                return jsonify({'success': False, 'error': 'Image not found'}), 404
            
            return send_file(image_path, mimetype='image/png')
        
        @self.app.route('/api/generative/text', methods=['POST'])
        def generate_text():
            """Generate text from a prompt"""
            data = request.get_json()
            
            if 'prompt' not in data:
                return jsonify({'success': False, 'error': 'Prompt is required'}), 400
            
            prompt = data['prompt']
            model = data.get('model', 'local')
            max_length = min(data.get('max_length', 500), 2000)  # Limit to 2000 tokens max
            temperature = data.get('temperature', 0.7)
            
            try:
                # Generate text
                generated_text, model_used = self._generate_text(prompt, model, max_length, temperature)
                
                # Save generated text
                text_id = str(uuid.uuid4())
                text_data = {
                    'id': text_id,
                    'prompt': prompt,
                    'text': generated_text,
                    'model': model_used,
                    'created_at': int(time.time())
                }
                
                text_path = os.path.join(self.texts_dir, f"{text_id}.json")
                with open(text_path, 'w') as f:
                    json.dump(text_data, f, indent=2)
                
                return jsonify({
                    'success': True,
                    'text_id': text_id,
                    'text': generated_text,
                    'model': model_used
                })
            
            except Exception as e:
                logger.error(f"Error generating text: {str(e)}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/generative/creative', methods=['POST'])
        def generate_creative_content():
            """Generate creative content (code, stories, etc.)"""
            data = request.get_json()
            
            if 'prompt' not in data or 'type' not in data:
                return jsonify({'success': False, 'error': 'Prompt and content type are required'}), 400
            
            prompt = data['prompt']
            content_type = data['type']
            model = data.get('model', 'default')
            
            # Validate content type
            valid_types = ['code', 'story', 'poem', 'script', 'marketing']
            if content_type not in valid_types:
                return jsonify({'success': False, 'error': f'Invalid content type. Must be one of: {", ".join(valid_types)}'}), 400
            
            try:
                # Generate creative content
                content, model_used = self._generate_creative_content(prompt, content_type, model)
                
                # Save generated content
                content_id = str(uuid.uuid4())
                content_data = {
                    'id': content_id,
                    'prompt': prompt,
                    'content': content,
                    'type': content_type,
                    'model': model_used,
                    'created_at': int(time.time())
                }
                
                content_path = os.path.join(self.creative_dir, f"{content_id}.json")
                with open(content_path, 'w') as f:
                    json.dump(content_data, f, indent=2)
                
                return jsonify({
                    'success': True,
                    'content_id': content_id,
                    'content': content,
                    'type': content_type,
                    'model': model_used
                })
            
            except Exception as e:
                logger.error(f"Error generating creative content: {str(e)}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/generative/enhance', methods=['POST'])
        def enhance_content():
            """Enhance existing content (improve text, upscale images, etc.)"""
            data = request.get_json()
            
            if 'content' not in data or 'type' not in data:
                return jsonify({'success': False, 'error': 'Content and type are required'}), 400
            
            content = data['content']
            content_type = data['type']
            enhancement = data.get('enhancement', 'quality')
            
            # Validate content type
            valid_types = ['text', 'image', 'code']
            if content_type not in valid_types:
                return jsonify({'success': False, 'error': f'Invalid content type. Must be one of: {", ".join(valid_types)}'}), 400
            
            try:
                # Enhance content
                enhanced_content, method_used = self._enhance_content(content, content_type, enhancement)
                
                return jsonify({
                    'success': True,
                    'enhanced_content': enhanced_content,
                    'type': content_type,
                    'enhancement': enhancement,
                    'method': method_used
                })
            
            except Exception as e:
                logger.error(f"Error enhancing content: {str(e)}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/generative/models', methods=['GET'])
        def get_available_models():
            """Get available generative AI models"""
            available_models = {
                'image': list(self.image_models.keys()) if self.image_models else ['api_fallback'],
                'text': list(self.text_models.keys()) if self.text_models else ['api_fallback'],
                'creative': list(self.creative_models.keys()) if self.creative_models else ['api_fallback']
            }
            
            return jsonify({
                'success': True,
                'models': available_models
            })
    
    def _generate_images(self, prompt: str, model: str = 'stable_diffusion', 
                         size: int = 512, num_images: int = 1) -> List[str]:
        """
        Generate images from a text prompt
        
        Args:
            prompt: Text prompt for image generation
            model: Model to use for generation
            size: Image size (square)
            num_images: Number of images to generate
            
        Returns:
            List of paths to generated images
        """
        image_paths = []
        
        # Use Stable Diffusion if available and selected
        if model == 'stable_diffusion' and 'stable_diffusion' in self.image_models and self.image_models['stable_diffusion']:
            # Generate images with Stable Diffusion
            images = self.image_models['stable_diffusion'](
                prompt=prompt,
                height=size,
                width=size,
                num_images_per_prompt=num_images
            ).images
            
            # Save images
            for i, image in enumerate(images):
                image_id = f"sd_{int(time.time())}_{i}.png"
                image_path = os.path.join(self.images_dir, image_id)
                image.save(image_path)
                image_paths.append(image_path)
        
        # Use DALL-E if available and selected
        elif model == 'dalle' and 'dalle' in self.image_models:
            import openai
            openai.api_key = self.image_models['dalle']['api_key']
            
            # Map size to DALL-E size format
            if size <= 512:
                dalle_size = "256x256"
            elif size <= 768:
                dalle_size = "512x512"
            else:
                dalle_size = "1024x1024"
            
            # Generate images with DALL-E
            for i in range(num_images):
                response = openai.images.generate(
                    model=self.image_models['dalle']['model'],
                    prompt=prompt,
                    size=dalle_size,
                    n=1
                )
                
                # Get image URL or base64 data
                image_url = response.data[0].url
                
                # Download image
                image_response = requests.get(image_url)
                image = Image.open(io.BytesIO(image_response.content))
                
                # Save image
                image_id = f"dalle_{int(time.time())}_{i}.png"
                image_path = os.path.join(self.images_dir, image_id)
                image.save(image_path)
                image_paths.append(image_path)
        
        # Fallback to Hugging Face API for image generation
        else:
            # This is a simplified implementation using Hugging Face API
            # In a real implementation, you would use the appropriate API endpoint
            
            # Generate a placeholder image for demonstration
            image = Image.new('RGB', (size, size), color='white')
            
            # Add text to the image
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(image)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                font = ImageFont.load_default()
            
            draw.text((10, 10), f"Generated Image\nPrompt: {prompt}", fill="black", font=font)
            
            # Save the image
            image_id = f"placeholder_{int(time.time())}.png"
            image_path = os.path.join(self.images_dir, image_id)
            image.save(image_path)
            image_paths.append(image_path)
            
            logger.warning(f"Using placeholder image generation for model: {model}")
        
        return image_paths
    
    def _generate_text(self, prompt: str, model: str = 'local', 
                      max_length: int = 500, temperature: float = 0.7) -> Tuple[str, str]:
        """
        Generate text from a prompt
        
        Args:
            prompt: Text prompt for generation
            model: Model to use for generation
            max_length: Maximum length of generated text
            temperature: Temperature for generation (higher = more creative)
            
        Returns:
            Tuple of (generated text, model used)
        """
        # Use local model if available and selected
        if model == 'local' and 'local' in self.text_models and self.text_models['local']:
            # Generate text with local model
            result = self.text_models['local'](
                prompt,
                max_length=max_length,
                temperature=temperature,
                num_return_sequences=1
            )
            
            # Extract generated text
            generated_text = result[0]['generated_text']
            
            return generated_text, 'local'
        
        # Use OpenAI if available and selected
        elif model == 'openai' and 'openai' in self.text_models:
            import openai
            openai.api_key = self.text_models['openai']['api_key']
            
            # Generate text with OpenAI
            response = openai.chat.completions.create(
                model=self.text_models['openai']['model'],
                messages=[
                    {"role": "system", "content": "You are a creative assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_length,
                temperature=temperature
            )
            
            # Extract generated text
            generated_text = response.choices[0].message.content
            
            return generated_text, 'openai'
        
        # Use Anthropic if available and selected
        elif model == 'anthropic' and 'anthropic' in self.text_models:
            # Generate text with Anthropic
            response = self.text_models['anthropic']['client'].messages.create(
                model=self.text_models['anthropic']['model'],
                max_tokens=max_length,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract generated text
            generated_text = response.content[0].text
            
            return generated_text, 'anthropic'
        
        # Fallback to simple text generation
        else:
            # This is a simplified implementation
            # In a real implementation, you would use an appropriate fallback
            
            generated_text = f"Generated text based on prompt: {prompt}\n\n"
            generated_text += "This is placeholder text. In a real implementation, this would be generated by an AI model."
            
            logger.warning(f"Using placeholder text generation for model: {model}")
            
            return generated_text, 'fallback'
    
    def _generate_creative_content(self, prompt: str, content_type: str, 
                                 model: str = 'default') -> Tuple[str, str]:
        """
        Generate creative content based on type and prompt
        
        Args:
            prompt: Text prompt for generation
            content_type: Type of content to generate (code, story, etc.)
            model: Model to use for generation
            
        Returns:
            Tuple of (generated content, model used)
        """
        # For code generation
        if content_type == 'code' and 'code' in self.creative_models and self.creative_models['code']:
            # Enhance prompt for code generation
            code_prompt = f"Write {prompt}\n\n```"
            
            # Generate code with code model
            result = self.creative_models['code'](
                code_prompt,
                max_length=1000,
                temperature=0.2,
                num_return_sequences=1
            )
            
            # Extract generated code
            generated_code = result[0]['generated_text']
            
            # Clean up the code
            if "```" in generated_code:
                # Extract code between backticks
                code_parts = generated_code.split("```")
                if len(code_parts) >= 3:
                    generated_code = code_parts[1]
                    if generated_code.startswith(("python", "javascript", "java")):
                        generated_code = generated_code[generated_code.find("\n")+1:]
            
            return generated_code, 'code_model'
        
        # For other creative content, use text generation
        else:
            # Enhance prompt based on content type
            enhanced_prompt = prompt
            
            if content_type == 'story':
                enhanced_prompt = f"Write a creative story about {prompt}. Include characters, setting, and plot."
            elif content_type == 'poem':
                enhanced_prompt = f"Write a poem about {prompt}. Be creative and use vivid imagery."
            elif content_type == 'script':
                enhanced_prompt = f"Write a script or dialogue about {prompt}. Include character names and stage directions."
            elif content_type == 'marketing':
                enhanced_prompt = f"Write marketing copy for {prompt}. Be persuasive and highlight benefits."
            elif content_type == 'code':
                enhanced_prompt = f"Write code for {prompt}. Include comments and explanations."
            
            # Use text generation for creative content
            generated_content, model_used = self._generate_text(
                enhanced_prompt,
                model='openai' if 'openai' in self.text_models else 'local',
                max_length=1000,
                temperature=0.8
            )
            
            return generated_content, model_used
    
    def _enhance_content(self, content: str, content_type: str, 
                       enhancement: str = 'quality') -> Tuple[str, str]:
        """
        Enhance existing content
        
        Args:
            content: Content to enhance
            content_type: Type of content (text, image, code)
            enhancement: Type of enhancement to apply
            
        Returns:
            Tuple of (enhanced content, method used)
        """
        # For text enhancement
        if content_type == 'text':
            # Create enhancement prompt based on enhancement type
            if enhancement == 'quality':
                prompt = f"Improve the quality of this text while preserving its meaning:\n\n{content}"
            elif enhancement == 'simplify':
                prompt = f"Simplify this text to make it easier to understand:\n\n{content}"
            elif enhancement == 'expand':
                prompt = f"Expand on this text with more details and examples:\n\n{content}"
            else:
                prompt = f"Enhance this text ({enhancement}):\n\n{content}"
            
            # Use text generation for enhancement
            enhanced_content, model_used = self._generate_text(
                prompt,
                model='openai' if 'openai' in self.text_models else 'local',
                max_length=len(content) * 2,
                temperature=0.3
            )
            
            return enhanced_content, f"text_{model_used}"
        
        # For code enhancement
        elif content_type == 'code':
            # Create enhancement prompt based on enhancement type
            if enhancement == 'quality':
                prompt = f"Improve this code while preserving its functionality:\n\n{content}"
            elif enhancement == 'optimize':
                prompt = f"Optimize this code for better performance:\n\n{content}"
            elif enhancement == 'document':
                prompt = f"Add detailed comments and documentation to this code:\n\n{content}"
            else:
                prompt = f"Enhance this code ({enhancement}):\n\n{content}"
            
            # Use code model or text generation for enhancement
            if 'code' in self.creative_models and self.creative_models['code']:
                result = self.creative_models['code'](
                    prompt,
                    max_length=len(content) * 2,
                    temperature=0.2,
                    num_return_sequences=1
                )
                
                enhanced_content = result[0]['generated_text']
                method_used = 'code_model'
            else:
                enhanced_content, model_used = self._generate_text(
                    prompt,
                    model='openai' if 'openai' in self.text_models else 'local',
                    max_length=len(content) * 2,
                    temperature=0.3
                )
                method_used = f"text_{model_used}"
            
            return enhanced_content, method_used
        
        # For image enhancement (placeholder implementation)
        elif content_type == 'image':
            # This is a simplified implementation
            # In a real implementation, you would use image enhancement models
            
            logger.warning("Image enhancement is not fully implemented")
            
            return content, 'image_placeholder'
        
        # Fallback for unsupported content types
        else:
            return content, 'no_enhancement'

# Function to initialize and register the generative AI features with the Flask app
def init_generative_ai_features(app: Flask, config: Dict[str, Any]) -> GenerativeAIManager:
    """Initialize and register generative AI features with the Flask app"""
    generative_ai_manager = GenerativeAIManager(app, config)
    return generative_ai_manager
