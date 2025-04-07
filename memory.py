"""
Memory Module for Manus Clone

This module handles conversation history and context management for the AI agent.
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Memory:
    """Memory class for managing conversation history and context"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Memory module with configuration
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.storage_dir = config.get("storage_dir", "/tmp/manus_clone_memory")
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # In-memory cache for active conversations
        self.conversation_cache = {}
        
        logger.info("Memory module initialized successfully")
    
    def add_user_message(self, message: str, conversation_id: str):
        """
        Add a user message to the conversation history
        
        Args:
            message: The user's message
            conversation_id: The conversation ID
        """
        self._add_message_to_conversation(message, "user", conversation_id)
    
    def add_assistant_message(self, message: str, conversation_id: str):
        """
        Add an assistant message to the conversation history
        
        Args:
            message: The assistant's message
            conversation_id: The conversation ID
        """
        self._add_message_to_conversation(message, "assistant", conversation_id)
    
    def _add_message_to_conversation(self, message: str, role: str, conversation_id: str):
        """
        Add a message to the conversation history
        
        Args:
            message: The message content
            role: The role of the message sender (user or assistant)
            conversation_id: The conversation ID
        """
        # Load conversation from cache or storage
        conversation = self.get_conversation_history(conversation_id)
        
        # Add the new message
        conversation.append({
            "role": role,
            "content": message,
            "timestamp": time.time()
        })
        
        # Update cache
        self.conversation_cache[conversation_id] = conversation
        
        # Save to storage
        self._save_conversation(conversation_id, conversation)
        
        logger.info(f"Added {role} message to conversation {conversation_id}")
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Get the conversation history for a given conversation ID
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            List of messages in the conversation
        """
        # Check if conversation is in cache
        if conversation_id in self.conversation_cache:
            return self.conversation_cache[conversation_id]
        
        # Load from storage
        conversation_file = os.path.join(self.storage_dir, f"{conversation_id}.json")
        if os.path.exists(conversation_file):
            try:
                with open(conversation_file, 'r', encoding='utf-8') as f:
                    conversation = json.load(f)
                
                # Update cache
                self.conversation_cache[conversation_id] = conversation
                
                return conversation
            except Exception as e:
                logger.error(f"Error loading conversation {conversation_id}: {str(e)}")
        
        # Return empty conversation if not found
        return []
    
    def _save_conversation(self, conversation_id: str, conversation: List[Dict[str, Any]]):
        """
        Save a conversation to storage
        
        Args:
            conversation_id: The conversation ID
            conversation: The conversation history
        """
        conversation_file = os.path.join(self.storage_dir, f"{conversation_id}.json")
        try:
            with open(conversation_file, 'w', encoding='utf-8') as f:
                json.dump(conversation, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving conversation {conversation_id}: {str(e)}")
    
    def get_all_conversations(self) -> List[str]:
        """
        Get a list of all conversation IDs
        
        Returns:
            List of conversation IDs
        """
        conversations = []
        try:
            for filename in os.listdir(self.storage_dir):
                if filename.endswith('.json'):
                    conversation_id = filename[:-5]  # Remove .json extension
                    conversations.append(conversation_id)
        except Exception as e:
            logger.error(f"Error listing conversations: {str(e)}")
        
        return conversations
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation
        
        Args:
            conversation_id: The conversation ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        conversation_file = os.path.join(self.storage_dir, f"{conversation_id}.json")
        try:
            if os.path.exists(conversation_file):
                os.remove(conversation_file)
                
                # Remove from cache if present
                if conversation_id in self.conversation_cache:
                    del self.conversation_cache[conversation_id]
                
                logger.info(f"Deleted conversation {conversation_id}")
                return True
            else:
                logger.warning(f"Conversation {conversation_id} not found for deletion")
                return False
        except Exception as e:
            logger.error(f"Error deleting conversation {conversation_id}: {str(e)}")
            return False
    
    def clear_all_conversations(self) -> bool:
        """
        Clear all conversations
        
        Returns:
            True if successful, False otherwise
        """
        try:
            for filename in os.listdir(self.storage_dir):
                if filename.endswith('.json'):
                    os.remove(os.path.join(self.storage_dir, filename))
            
            # Clear cache
            self.conversation_cache = {}
            
            logger.info("Cleared all conversations")
            return True
        except Exception as e:
            logger.error(f"Error clearing conversations: {str(e)}")
            return False
    
    def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get a summary of a conversation
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            Dictionary containing conversation summary
        """
        conversation = self.get_conversation_history(conversation_id)
        
        if not conversation:
            return {
                "id": conversation_id,
                "message_count": 0,
                "last_updated": None,
                "summary": "محادثة فارغة"
            }
        
        # Get the timestamp of the last message
        last_updated = max(msg.get("timestamp", 0) for msg in conversation)
        
        # Count messages by role
        user_messages = sum(1 for msg in conversation if msg.get("role") == "user")
        assistant_messages = sum(1 for msg in conversation if msg.get("role") == "assistant")
        
        # Get the first user message as a simple summary
        first_user_message = next((msg.get("content", "") for msg in conversation if msg.get("role") == "user"), "")
        summary = first_user_message[:100] + "..." if len(first_user_message) > 100 else first_user_message
        
        return {
            "id": conversation_id,
            "message_count": len(conversation),
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "last_updated": last_updated,
            "summary": summary
        }
