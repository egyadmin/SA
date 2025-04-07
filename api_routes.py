"""
API Routes for AI Model Integration

This module provides API endpoints for:
- Testing connection to AI models
- Saving and retrieving API keys
- Managing AI model settings
"""

import os
import json
import logging
from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash

# Import AI models integration
from ai_models_integration import AIModelsIntegration

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create blueprint
ai_api = Blueprint('ai_api', __name__)

# Initialize AI models integration
config = {}
ai_integration = AIModelsIntegration(config)

# Secure storage for API keys (in-memory for demo, should use secure storage in production)
api_keys = {}

@ai_api.route('/api/ai/test-connection', methods=['POST'])
def test_connection():
    """Test connection to AI model API"""
    try:
        data = request.json
        provider = data.get('provider', '').lower()
        api_key = data.get('apiKey', '')
        
        if not provider or not api_key:
            return jsonify({'success': False, 'error': 'Provider and API key are required'})
        
        # Validate API key format
        if provider == 'claude' and not api_key.startswith('sk-ant-'):
            return jsonify({'success': False, 'error': 'Invalid Claude API key format. Must start with sk-ant-'})
        elif provider == 'openai' and not api_key.startswith('sk-'):
            return jsonify({'success': False, 'error': 'Invalid OpenAI API key format. Must start with sk-'})
        
        # Test connection
        is_valid = ai_integration.validate_api_key(provider, api_key)
        
        if is_valid:
            # Save API key securely if connection is successful
            api_keys[provider] = api_key
            ai_integration.set_api_key(provider, api_key)
            
            return jsonify({'success': True, 'message': f'Successfully connected to {provider}'})
        else:
            return jsonify({'success': False, 'error': f'Failed to connect to {provider}. Invalid API key or service unavailable.'})
    
    except Exception as e:
        logger.error(f"Error testing connection: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@ai_api.route('/api/settings/save-api-key', methods=['POST'])
def save_api_key():
    """Save API key for a provider"""
    try:
        data = request.json
        provider = data.get('provider', '').lower()
        api_key = data.get('apiKey', '')
        
        if not provider or not api_key:
            return jsonify({'success': False, 'error': 'Provider and API key are required'})
        
        # Validate API key format
        if provider == 'claude' and not api_key.startswith('sk-ant-'):
            return jsonify({'success': False, 'error': 'Invalid Claude API key format. Must start with sk-ant-'})
        elif provider == 'openai' and not api_key.startswith('sk-'):
            return jsonify({'success': False, 'error': 'Invalid OpenAI API key format. Must start with sk-'})
        
        # Save API key securely
        api_keys[provider] = api_key
        
        # Update AI integration
        ai_integration.set_api_key(provider, api_key)
        
        return jsonify({'success': True, 'message': f'API key for {provider} saved successfully'})
    
    except Exception as e:
        logger.error(f"Error saving API key: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@ai_api.route('/api/settings/get-api-key', methods=['GET'])
def get_api_key():
    """Get API key for a provider"""
    try:
        provider = request.args.get('provider', '').lower()
        
        if not provider:
            return jsonify({'success': False, 'error': 'Provider is required'})
        
        # Get API key
        api_key = api_keys.get(provider, '')
        
        # Mask API key for security (return only first and last 4 characters)
        masked_key = ''
        if api_key:
            if len(api_key) > 8:
                masked_key = api_key[:4] + '*' * (len(api_key) - 8) + api_key[-4:]
            else:
                masked_key = api_key
        
        return jsonify({'success': True, 'apiKey': masked_key})
    
    except Exception as e:
        logger.error(f"Error getting API key: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@ai_api.route('/api/ai/models', methods=['GET'])
def get_models():
    """Get available AI models"""
    try:
        models = ai_integration.get_available_models()
        return jsonify({'success': True, 'models': models})
    
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@ai_api.route('/api/ai/status', methods=['GET'])
def get_status():
    """Get status of AI model providers"""
    try:
        status = ai_integration.get_model_status()
        return jsonify({'success': True, 'status': status})
    
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})
