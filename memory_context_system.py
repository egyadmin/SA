"""
Memory and Context System Module for Manus Clone

This module handles long-term memory and context management including:
- Session memory and persistence
- Conversation history management
- Knowledge base integration
- Context window optimization
- Memory retrieval and search
"""

import os
import json
import time
import uuid
import logging
import datetime
import sqlite3
import pickle
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryAndContextSystem:
    """Memory and Context System class for improved information retention"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Memory and Context System with configuration
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.memory_dir = config.get('MEMORY_FOLDER', os.path.join(os.path.expanduser('~'), 'manus_clone', 'memory'))
        self.db_path = os.path.join(self.memory_dir, 'memory.db')
        self.max_conversation_length = config.get('MAX_CONVERSATION_LENGTH', 100)
        self.max_context_tokens = config.get('MAX_CONTEXT_TOKENS', 8000)
        self.current_session_id = str(uuid.uuid4())
        self.current_conversation = []
        
        # Ensure memory directory exists
        os.makedirs(self.memory_dir, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        logger.info("Memory and Context System initialized successfully")
        logger.info(f"Current session ID: {self.current_session_id}")
    
    def _init_database(self):
        """
        Initialize the SQLite database for memory storage
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create sessions table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                name TEXT,
                start_time TEXT,
                end_time TEXT,
                metadata TEXT
            )
            ''')
            
            # Create conversations table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                timestamp TEXT,
                role TEXT,
                content TEXT,
                metadata TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions (id)
            )
            ''')
            
            # Create knowledge table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge (
                id TEXT PRIMARY KEY,
                source TEXT,
                content TEXT,
                embedding BLOB,
                metadata TEXT,
                timestamp TEXT
            )
            ''')
            
            # Create facts table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                fact TEXT,
                source TEXT,
                confidence REAL,
                timestamp TEXT,
                metadata TEXT
            )
            ''')
            
            # Create entities table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT,
                type TEXT,
                properties TEXT,
                timestamp TEXT
            )
            ''')
            
            # Create relationships table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS relationships (
                id TEXT PRIMARY KEY,
                entity1_id TEXT,
                entity2_id TEXT,
                type TEXT,
                properties TEXT,
                timestamp TEXT,
                FOREIGN KEY (entity1_id) REFERENCES entities (id),
                FOREIGN KEY (entity2_id) REFERENCES entities (id)
            )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON conversations (session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_knowledge_source ON knowledge (source)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_entities_type ON entities (type)')
            
            # Start a new session
            cursor.execute(
                'INSERT INTO sessions (id, name, start_time, metadata) VALUES (?, ?, ?, ?)',
                (self.current_session_id, f"Session {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                 datetime.datetime.now().isoformat(), json.dumps({}))
            )
            
            conn.commit()
            conn.close()
            
            logger.info("Memory database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing memory database: {str(e)}")
            raise
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a message to the current conversation
        
        Args:
            role: Role of the message sender (user, assistant, system)
            content: Message content
            metadata: Additional metadata for the message
            
        Returns:
            ID of the added message
        """
        message_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        
        # Add to current conversation
        message = {
            "id": message_id,
            "session_id": self.current_session_id,
            "timestamp": timestamp,
            "role": role,
            "content": content,
            "metadata": metadata or {}
        }
        self.current_conversation.append(message)
        
        # Trim conversation if needed
        if len(self.current_conversation) > self.max_conversation_length:
            self.current_conversation = self.current_conversation[-self.max_conversation_length:]
        
        # Save to database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                'INSERT INTO conversations (id, session_id, timestamp, role, content, metadata) VALUES (?, ?, ?, ?, ?, ?)',
                (message_id, self.current_session_id, timestamp, role, content, json.dumps(metadata or {}))
            )
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error saving message to database: {str(e)}")
        
        return message_id
    
    def get_conversation_history(self, 
                                session_id: Optional[str] = None, 
                                limit: Optional[int] = None,
                                include_metadata: bool = False) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session
        
        Args:
            session_id: Session ID (if None, uses current session)
            limit: Maximum number of messages to return
            include_metadata: Whether to include metadata in the results
            
        Returns:
            List of conversation messages
        """
        session_id = session_id or self.current_session_id
        limit = limit or self.max_conversation_length
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                'SELECT id, timestamp, role, content, metadata FROM conversations WHERE session_id = ? ORDER BY timestamp ASC LIMIT ?',
                (session_id, limit)
            )
            
            rows = cursor.fetchall()
            conn.close()
            
            messages = []
            for row in rows:
                message = {
                    "id": row[0],
                    "timestamp": row[1],
                    "role": row[2],
                    "content": row[3]
                }
                
                if include_metadata:
                    message["metadata"] = json.loads(row[4])
                
                messages.append(message)
            
            return messages
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {str(e)}")
            return []
    
    def get_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get list of sessions
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List of session information
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                'SELECT id, name, start_time, end_time, metadata FROM sessions ORDER BY start_time DESC LIMIT ?',
                (limit,)
            )
            
            rows = cursor.fetchall()
            
            # Get message counts for each session
            sessions = []
            for row in rows:
                session_id = row[0]
                
                cursor.execute('SELECT COUNT(*) FROM conversations WHERE session_id = ?', (session_id,))
                message_count = cursor.fetchone()[0]
                
                session = {
                    "id": session_id,
                    "name": row[1],
                    "start_time": row[2],
                    "end_time": row[3],
                    "message_count": message_count,
                    "metadata": json.loads(row[4])
                }
                
                sessions.append(session)
            
            conn.close()
            return sessions
        except Exception as e:
            logger.error(f"Error retrieving sessions: {str(e)}")
            return []
    
    def start_new_session(self, name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Start a new session
        
        Args:
            name: Name for the new session
            metadata: Additional metadata for the session
            
        Returns:
            ID of the new session
        """
        # End current session
        self.end_session()
        
        # Create new session
        self.current_session_id = str(uuid.uuid4())
        self.current_conversation = []
        
        timestamp = datetime.datetime.now().isoformat()
        session_name = name or f"Session {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                'INSERT INTO sessions (id, name, start_time, metadata) VALUES (?, ?, ?, ?)',
                (self.current_session_id, session_name, timestamp, json.dumps(metadata or {}))
            )
            
            conn.commit()
            conn.close()
            
            logger.info(f"Started new session: {self.current_session_id}")
        except Exception as e:
            logger.error(f"Error starting new session: {str(e)}")
        
        return self.current_session_id
    
    def end_session(self) -> bool:
        """
        End the current session
        
        Returns:
            True if successful, False otherwise
        """
        if not self.current_session_id:
            return False
        
        timestamp = datetime.datetime.now().isoformat()
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                'UPDATE sessions SET end_time = ? WHERE id = ?',
                (timestamp, self.current_session_id)
            )
            
            conn.commit()
            conn.close()
            
            logger.info(f"Ended session: {self.current_session_id}")
            return True
        except Exception as e:
            logger.error(f"Error ending session: {str(e)}")
            return False
    
    def switch_session(self, session_id: str) -> bool:
        """
        Switch to an existing session
        
        Args:
            session_id: ID of the session to switch to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if session exists
            cursor.execute('SELECT id FROM sessions WHERE id = ?', (session_id,))
            if not cursor.fetchone():
                conn.close()
                logger.error(f"Session not found: {session_id}")
                return False
            
            # End current session
            self.end_session()
            
            # Switch to new session
            self.current_session_id = session_id
            
            # Load conversation history
            self.current_conversation = []
            
            cursor.execute(
                'SELECT id, timestamp, role, content, metadata FROM conversations WHERE session_id = ? ORDER BY timestamp ASC',
                (session_id,)
            )
            
            rows = cursor.fetchall()
            for row in rows:
                message = {
                    "id": row[0],
                    "session_id": session_id,
                    "timestamp": row[1],
                    "role": row[2],
                    "content": row[3],
                    "metadata": json.loads(row[4])
                }
                self.current_conversation.append(message)
            
            conn.close()
            
            logger.info(f"Switched to session: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Error switching session: {str(e)}")
            return False
    
    def rename_session(self, session_id: str, name: str) -> bool:
        """
        Rename a session
        
        Args:
            session_id: ID of the session to rename
            name: New name for the session
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('UPDATE sessions SET name = ? WHERE id = ?', (name, session_id))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Renamed session {session_id} to '{name}'")
            return True
        except Exception as e:
            logger.error(f"Error renaming session: {str(e)}")
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and its conversations
        
        Args:
            session_id: ID of the session to delete
            
        Returns:
            True if successful, False otherwise
        """
        if session_id == self.current_session_id:
            logger.error("Cannot delete current session")
            return False
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete conversations first (foreign key constraint)
            cursor.execute('DELETE FROM conversations WHERE session_id = ?', (session_id,))
            
            # Delete session
            cursor.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Deleted session: {session_id}")
            return Tr
(Content truncated due to size limit. Use line ranges to read in chunks)