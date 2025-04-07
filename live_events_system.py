"""
Live Events System Module for Manus Clone

This module handles real-time event tracking and display including:
- Event creation and management
- Event categorization and prioritization
- Real-time event streaming
- Event history and persistence
"""

import os
import json
import time
import uuid
import logging
import datetime
import threading
import queue
from typing import Dict, List, Any, Optional, Union, Callable

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LiveEventsSystem:
    """Live Events System class for real-time event tracking and display"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Live Events System with configuration
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.events_dir = config.get('EVENTS_FOLDER', os.path.join(os.path.expanduser('~'), 'manus_clone', 'events'))
        self.max_events = config.get('MAX_EVENTS', 1000)
        self.events = []
        self.event_listeners = []
        self.event_queue = queue.Queue()
        self.event_thread = None
        self.running = False
        
        # Ensure events directory exists
        os.makedirs(self.events_dir, exist_ok=True)
        
        # Load existing events if available
        self._load_events()
        
        logger.info("Live Events System initialized successfully")
    
    def start(self):
        """
        Start the event processing thread
        """
        if not self.running:
            self.running = True
            self.event_thread = threading.Thread(target=self._process_events)
            self.event_thread.daemon = True
            self.event_thread.start()
            logger.info("Event processing thread started")
    
    def stop(self):
        """
        Stop the event processing thread
        """
        if self.running:
            self.running = False
            if self.event_thread:
                self.event_thread.join(timeout=2.0)
            logger.info("Event processing thread stopped")
    
    def add_event(self, 
                 event_type: str, 
                 content: str, 
                 category: str = "info", 
                 source: str = "system",
                 metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Add a new event to the system
        
        Args:
            event_type: Type of event
            content: Event content or message
            category: Event category (info, success, warning, error)
            source: Source of the event
            metadata: Additional metadata for the event
            
        Returns:
            The created event object
        """
        # Create event object
        event = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.now().isoformat(),
            "type": event_type,
            "content": content,
            "category": category,
            "source": source,
            "metadata": metadata or {}
        }
        
        # Add to queue for processing
        self.event_queue.put(event)
        
        return event
    
    def get_events(self, 
                  limit: Optional[int] = None, 
                  event_type: Optional[str] = None,
                  category: Optional[str] = None,
                  source: Optional[str] = None,
                  start_time: Optional[str] = None,
                  end_time: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get events with optional filtering
        
        Args:
            limit: Maximum number of events to return
            event_type: Filter by event type
            category: Filter by category
            source: Filter by source
            start_time: Filter by start time (ISO format)
            end_time: Filter by end time (ISO format)
            
        Returns:
            List of filtered events
        """
        filtered_events = self.events.copy()
        
        # Apply filters
        if event_type:
            filtered_events = [e for e in filtered_events if e["type"] == event_type]
        
        if category:
            filtered_events = [e for e in filtered_events if e["category"] == category]
        
        if source:
            filtered_events = [e for e in filtered_events if e["source"] == source]
        
        if start_time:
            filtered_events = [e for e in filtered_events if e["timestamp"] >= start_time]
        
        if end_time:
            filtered_events = [e for e in filtered_events if e["timestamp"] <= end_time]
        
        # Sort by timestamp (newest first)
        filtered_events.sort(key=lambda e: e["timestamp"], reverse=True)
        
        # Apply limit
        if limit:
            filtered_events = filtered_events[:limit]
        
        return filtered_events
    
    def get_event_by_id(self, event_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an event by its ID
        
        Args:
            event_id: ID of the event to retrieve
            
        Returns:
            Event object or None if not found
        """
        for event in self.events:
            if event["id"] == event_id:
                return event
        return None
    
    def clear_events(self, older_than: Optional[str] = None) -> int:
        """
        Clear events from memory
        
        Args:
            older_than: Clear events older than this timestamp (ISO format)
            
        Returns:
            Number of events cleared
        """
        if older_than:
            original_count = len(self.events)
            self.events = [e for e in self.events if e["timestamp"] >= older_than]
            return original_count - len(self.events)
        else:
            count = len(self.events)
            self.events = []
            return count
    
    def add_event_listener(self, callback: Callable[[Dict[str, Any]], None]) -> str:
        """
        Add an event listener to receive real-time events
        
        Args:
            callback: Function to call when a new event is processed
            
        Returns:
            Listener ID
        """
        listener_id = str(uuid.uuid4())
        self.event_listeners.append({
            "id": listener_id,
            "callback": callback
        })
        return listener_id
    
    def remove_event_listener(self, listener_id: str) -> bool:
        """
        Remove an event listener
        
        Args:
            listener_id: ID of the listener to remove
            
        Returns:
            True if removed, False if not found
        """
        original_count = len(self.event_listeners)
        self.event_listeners = [l for l in self.event_listeners if l["id"] != listener_id]
        return len(self.event_listeners) < original_count
    
    def save_events(self, file_path: Optional[str] = None) -> str:
        """
        Save events to a file
        
        Args:
            file_path: Path to save events to (if None, uses default path)
            
        Returns:
            Path where events were saved
        """
        if not file_path:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(self.events_dir, f"events_{timestamp}.json")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.events, f, ensure_ascii=False, indent=2)
        
        return file_path
    
    def load_events_from_file(self, file_path: str) -> int:
        """
        Load events from a file
        
        Args:
            file_path: Path to load events from
            
        Returns:
            Number of events loaded
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded_events = json.load(f)
            
            # Validate events
            valid_events = []
            for event in loaded_events:
                if isinstance(event, dict) and "id" in event and "timestamp" in event:
                    valid_events.append(event)
            
            # Add to existing events
            self.events.extend(valid_events)
            
            # Ensure we don't exceed max events
            if len(self.events) > self.max_events:
                self.events = self.events[-self.max_events:]
            
            return len(valid_events)
        except Exception as e:
            logger.error(f"Error loading events from file: {str(e)}")
            return 0
    
    def get_event_stats(self) -> Dict[str, Any]:
        """
        Get statistics about events
        
        Returns:
            Dictionary with event statistics
        """
        if not self.events:
            return {
                "total": 0,
                "categories": {},
                "types": {},
                "sources": {}
            }
        
        # Count by category
        categories = {}
        for event in self.events:
            category = event["category"]
            categories[category] = categories.get(category, 0) + 1
        
        # Count by type
        types = {}
        for event in self.events:
            event_type = event["type"]
            types[event_type] = types.get(event_type, 0) + 1
        
        # Count by source
        sources = {}
        for event in self.events:
            source = event["source"]
            sources[source] = sources.get(source, 0) + 1
        
        # Get time range
        timestamps = [event["timestamp"] for event in self.events]
        timestamps.sort()
        
        return {
            "total": len(self.events),
            "categories": categories,
            "types": types,
            "sources": sources,
            "oldest": timestamps[0] if timestamps else None,
            "newest": timestamps[-1] if timestamps else None
        }
    
    def _process_events(self):
        """
        Process events from the queue (runs in a separate thread)
        """
        while self.running:
            try:
                # Get event from queue with timeout
                try:
                    event = self.event_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Add to events list
                self.events.append(event)
                
                # Ensure we don't exceed max events
                if len(self.events) > self.max_events:
                    self.events.pop(0)
                
                # Notify listeners
                for listener in self.event_listeners:
                    try:
                        listener["callback"](event)
                    except Exception as e:
                        logger.error(f"Error in event listener: {str(e)}")
                
                # Mark as done
                self.event_queue.task_done()
                
                # Periodically save events
                if len(self.events) % 100 == 0:
                    self._save_events()
            except Exception as e:
                logger.error(f"Error processing event: {str(e)}")
    
    def _load_events(self):
        """
        Load events from the most recent events file
        """
        try:
            # Find the most recent events file
            event_files = [f for f in os.listdir(self.events_dir) if f.startswith("events_") and f.endswith(".json")]
            if not event_files:
                return
            
            # Sort by timestamp (newest first)
            event_files.sort(reverse=True)
            
            # Load events from the most recent file
            latest_file = os.path.join(self.events_dir, event_files[0])
            self.load_events_from_file(latest_file)
            
            logger.info(f"Loaded {len(self.events)} events from {latest_file}")
        except Exception as e:
            logger.error(f"Error loading events: {str(e)}")
    
    def _save_events(self):
        """
        Save events to a file
        """
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(self.events_dir, f"events_{timestamp}.json")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.events, f, ensure_ascii=False, indent=2)
            
            # Clean up old event files
            self._cleanup_event_files()
            
            logger.info(f"Saved {len(self.events)} events to {file_path}")
        except Exception as e:
            logger.error(f"Error saving events: {str(e)}")
    
    def _cleanup_event_files(self, max_files: int = 10):
        """
        Clean up old event files
        
        Args:
            max_files: Maximum number of event files to keep
        """
        try:
            event_files = [f for f in os.listdir(self.events_dir) if f.startswith("events_") and f.endswith(".json")]
            if len(event_files) <= max_files:
                return
            
            # Sort by timestamp (oldest first)
            event_files.sort()
            
            # Delete oldest files
            for file_name in event_files[:-max_files]:
                file_path = os.path.join(self.events_dir, file_name)
                os.remove(file_path)
                logger.info(f"Deleted old event file: {file_path}")
        except Exception as e:
            logger.error(f"Error cleaning up event files: {str(e)}")
    
    def __del__(self):
        """
        Clean up resources when the object is destroyed
        """
        self.stop()
        self._save_events()
