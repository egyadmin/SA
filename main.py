from flask import Flask, render_template, request, jsonify
import os
import json
import time
import uuid
from werkzeug.utils import secure_filename

from agent.core import AIAgent
from config.config import Config

# Initialize the application
app = Flask(__name__)
app.config.from_object(Config)

# Initialize the AI agent
agent = AIAgent(app.config)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CONVERSATION_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    """Render the main application page"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Process a chat message and return the response"""
    # Get the message from the request
    message = request.form.get('message', '')
    conversation_id = request.form.get('conversation_id', str(uuid.uuid4()))
    
    # Handle file uploads if any
    files = []
    if 'files' in request.files:
        uploaded_files = request.files.getlist('files')
        for file in uploaded_files:
            if file.filename:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                files.append(file_path)
    
    # Process the message
    response, status_updates, tools_used, attachments = agent.process_request(
        message, 
        conversation_id=conversation_id,
        files=files
    )
    
    # Return the response
    return jsonify({
        'response': response,
        'conversation_id': conversation_id,
        'status_updates': status_updates,
        'tools_used': tools_used,
        'attachments': attachments
    })

@app.route('/api/status', methods=['GET'])
def status():
    """Get the current status of a conversation"""
    conversation_id = request.args.get('conversation_id', '')
    
    # Get the status updates for the conversation
    status_updates = agent.get_status_updates(conversation_id)
    tools_used = agent.get_tools_used(conversation_id)
    
    # Return the status updates
    return jsonify({
        'conversation_id': conversation_id,
        'status_updates': status_updates,
        'tools_used': tools_used
    })

@app.route('/api/conversations', methods=['GET'])
def get_conversations():
    """Get all conversations"""
    conversations = []
    
    # Get all conversation files
    for filename in os.listdir(app.config['CONVERSATION_FOLDER']):
        if filename.endswith('.json'):
            conversation_id = filename[:-5]  # Remove .json extension
            conversation_path = os.path.join(app.config['CONVERSATION_FOLDER'], filename)
            
            try:
                with open(conversation_path, 'r', encoding='utf-8') as f:
                    conversation_data = json.load(f)
                
                # Get the first user message as a summary
                summary = ''
                for message in conversation_data:
                    if message.get('role') == 'user':
                        summary = message.get('content', '')
                        break
                
                # Truncate summary if too long
                if len(summary) > 100:
                    summary = summary[:100] + '...'
                
                # Get the timestamp of the last message
                last_updated = max(msg.get('timestamp', 0) for msg in conversation_data) if conversation_data else 0
                
                conversations.append({
                    'id': conversation_id,
                    'summary': summary,
                    'last_updated': last_updated,
                    'message_count': len(conversation_data)
                })
            except Exception as e:
                app.logger.error(f"Error loading conversation {conversation_id}: {str(e)}")
    
    # Sort conversations by last updated timestamp (newest first)
    conversations.sort(key=lambda x: x.get('last_updated', 0), reverse=True)
    
    return jsonify({
        'conversations': conversations
    })

@app.route('/api/conversations/<conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    """Get a specific conversation"""
    conversation_path = os.path.join(app.config['CONVERSATION_FOLDER'], f"{conversation_id}.json")
    
    if os.path.exists(conversation_path):
        try:
            with open(conversation_path, 'r', encoding='utf-8') as f:
                conversation_data = json.load(f)
            
            return jsonify({
                'conversation_id': conversation_id,
                'messages': conversation_data
            })
        except Exception as e:
            app.logger.error(f"Error loading conversation {conversation_id}: {str(e)}")
            return jsonify({
                'error': f"Error loading conversation: {str(e)}"
            }), 500
    else:
        return jsonify({
            'conversation_id': conversation_id,
            'messages': []
        })

@app.route('/api/conversations/<conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id):
    """Delete a specific conversation"""
    conversation_path = os.path.join(app.config['CONVERSATION_FOLDER'], f"{conversation_id}.json")
    
    if os.path.exists(conversation_path):
        try:
            os.remove(conversation_path)
            return jsonify({
                'success': True,
                'message': f"Conversation {conversation_id} deleted successfully"
            })
        except Exception as e:
            app.logger.error(f"Error deleting conversation {conversation_id}: {str(e)}")
            return jsonify({
                'success': False,
                'error': f"Error deleting conversation: {str(e)}"
            }), 500
    else:
        return jsonify({
            'success': False,
            'error': f"Conversation {conversation_id} not found"
        }), 404

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Upload files"""
    if 'files' not in request.files:
        return jsonify({
            'success': False,
            'error': "No files provided"
        }), 400
    
    uploaded_files = request.files.getlist('files')
    saved_files = []
    
    for file in uploaded_files:
        if file.filename:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            saved_files.append({
                'name': filename,
                'path': file_path,
                'size': os.path.getsize(file_path),
                'type': file.content_type
            })
    
    return jsonify({
        'success': True,
        'files': saved_files
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
