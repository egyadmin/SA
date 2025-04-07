"""
Todo System Module for Manus Clone

This module handles the Todo system functionality including:
- Creating and managing todo lists
- Tracking task progress
- Organizing tasks by project
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TodoSystem:
    """Todo System class for managing tasks and todo lists"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Todo System with configuration
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.todo_dir = config.get('TODO_FOLDER', os.path.join(os.path.expanduser('~'), 'manus_clone', 'todos'))
        
        # Ensure todo directory exists
        os.makedirs(self.todo_dir, exist_ok=True)
        
        logger.info("Todo System initialized successfully")
    
    def create_todo_list(self, title: str, description: str = "", project: str = "default") -> Dict[str, Any]:
        """
        Create a new todo list
        
        Args:
            title: Title of the todo list
            description: Description of the todo list
            project: Project name for organizing todo lists
            
        Returns:
            Dictionary containing the created todo list information
        """
        try:
            # Generate a unique ID for the todo list
            todo_id = f"todo_{int(time.time())}"
            
            # Create the todo list structure
            todo_list = {
                "id": todo_id,
                "title": title,
                "description": description,
                "project": project,
                "created_at": time.time(),
                "updated_at": time.time(),
                "tasks": [],
                "completed": False
            }
            
            # Save the todo list
            self._save_todo_list(todo_list)
            
            return {
                "todo_list": todo_list,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error creating todo list: {str(e)}")
            return {"error": f"حدث خطأ أثناء إنشاء قائمة المهام: {str(e)}"}
    
    def add_task(self, todo_id: str, task: str, priority: str = "medium", due_date: Optional[float] = None) -> Dict[str, Any]:
        """
        Add a task to a todo list
        
        Args:
            todo_id: ID of the todo list
            task: Task description
            priority: Task priority (low, medium, high)
            due_date: Due date timestamp
            
        Returns:
            Dictionary containing the updated todo list
        """
        try:
            # Load the todo list
            todo_list = self._load_todo_list(todo_id)
            if not todo_list:
                return {"error": f"قائمة المهام غير موجودة: {todo_id}"}
            
            # Generate a unique ID for the task
            task_id = f"task_{int(time.time())}_{len(todo_list['tasks'])}"
            
            # Create the task structure
            task_item = {
                "id": task_id,
                "task": task,
                "priority": priority,
                "due_date": due_date,
                "created_at": time.time(),
                "completed": False,
                "completed_at": None
            }
            
            # Add the task to the todo list
            todo_list["tasks"].append(task_item)
            todo_list["updated_at"] = time.time()
            
            # Save the updated todo list
            self._save_todo_list(todo_list)
            
            return {
                "todo_list": todo_list,
                "task": task_item,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error adding task: {str(e)}")
            return {"error": f"حدث خطأ أثناء إضافة المهمة: {str(e)}"}
    
    def complete_task(self, todo_id: str, task_id: str) -> Dict[str, Any]:
        """
        Mark a task as completed
        
        Args:
            todo_id: ID of the todo list
            task_id: ID of the task
            
        Returns:
            Dictionary containing the updated todo list
        """
        try:
            # Load the todo list
            todo_list = self._load_todo_list(todo_id)
            if not todo_list:
                return {"error": f"قائمة المهام غير موجودة: {todo_id}"}
            
            # Find the task
            task_found = False
            for task in todo_list["tasks"]:
                if task["id"] == task_id:
                    task["completed"] = True
                    task["completed_at"] = time.time()
                    task_found = True
                    break
            
            if not task_found:
                return {"error": f"المهمة غير موجودة: {task_id}"}
            
            # Check if all tasks are completed
            all_completed = all(task["completed"] for task in todo_list["tasks"])
            if all_completed:
                todo_list["completed"] = True
            
            todo_list["updated_at"] = time.time()
            
            # Save the updated todo list
            self._save_todo_list(todo_list)
            
            return {
                "todo_list": todo_list,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error completing task: {str(e)}")
            return {"error": f"حدث خطأ أثناء إكمال المهمة: {str(e)}"}
    
    def update_task(self, todo_id: str, task_id: str, task: Optional[str] = None, 
                   priority: Optional[str] = None, due_date: Optional[float] = None) -> Dict[str, Any]:
        """
        Update a task in a todo list
        
        Args:
            todo_id: ID of the todo list
            task_id: ID of the task
            task: New task description (optional)
            priority: New task priority (optional)
            due_date: New due date timestamp (optional)
            
        Returns:
            Dictionary containing the updated todo list
        """
        try:
            # Load the todo list
            todo_list = self._load_todo_list(todo_id)
            if not todo_list:
                return {"error": f"قائمة المهام غير موجودة: {todo_id}"}
            
            # Find the task
            task_found = False
            for task_item in todo_list["tasks"]:
                if task_item["id"] == task_id:
                    if task is not None:
                        task_item["task"] = task
                    if priority is not None:
                        task_item["priority"] = priority
                    if due_date is not None:
                        task_item["due_date"] = due_date
                    
                    task_found = True
                    break
            
            if not task_found:
                return {"error": f"المهمة غير موجودة: {task_id}"}
            
            todo_list["updated_at"] = time.time()
            
            # Save the updated todo list
            self._save_todo_list(todo_list)
            
            return {
                "todo_list": todo_list,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error updating task: {str(e)}")
            return {"error": f"حدث خطأ أثناء تحديث المهمة: {str(e)}"}
    
    def delete_task(self, todo_id: str, task_id: str) -> Dict[str, Any]:
        """
        Delete a task from a todo list
        
        Args:
            todo_id: ID of the todo list
            task_id: ID of the task
            
        Returns:
            Dictionary containing the updated todo list
        """
        try:
            # Load the todo list
            todo_list = self._load_todo_list(todo_id)
            if not todo_list:
                return {"error": f"قائمة المهام غير موجودة: {todo_id}"}
            
            # Find and remove the task
            task_found = False
            for i, task in enumerate(todo_list["tasks"]):
                if task["id"] == task_id:
                    todo_list["tasks"].pop(i)
                    task_found = True
                    break
            
            if not task_found:
                return {"error": f"المهمة غير موجودة: {task_id}"}
            
            todo_list["updated_at"] = time.time()
            
            # Save the updated todo list
            self._save_todo_list(todo_list)
            
            return {
                "todo_list": todo_list,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error deleting task: {str(e)}")
            return {"error": f"حدث خطأ أثناء حذف المهمة: {str(e)}"}
    
    def get_todo_list(self, todo_id: str) -> Dict[str, Any]:
        """
        Get a todo list by ID
        
        Args:
            todo_id: ID of the todo list
            
        Returns:
            Dictionary containing the todo list
        """
        try:
            # Load the todo list
            todo_list = self._load_todo_list(todo_id)
            if not todo_list:
                return {"error": f"قائمة المهام غير موجودة: {todo_id}"}
            
            return {
                "todo_list": todo_list,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error getting todo list: {str(e)}")
            return {"error": f"حدث خطأ أثناء الحصول على قائمة المهام: {str(e)}"}
    
    def get_all_todo_lists(self, project: Optional[str] = None) -> Dict[str, Any]:
        """
        Get all todo lists, optionally filtered by project
        
        Args:
            project: Project name to filter by (optional)
            
        Returns:
            Dictionary containing the todo lists
        """
        try:
            todo_lists = []
            
            # Get all todo list files
            for filename in os.listdir(self.todo_dir):
                if filename.endswith('.json'):
                    todo_id = filename[:-5]  # Remove .json extension
                    todo_list = self._load_todo_list(todo_id)
                    
                    if todo_list:
                        # Filter by project if specified
                        if project is None or todo_list.get("project") == project:
                            todo_lists.append(todo_list)
            
            # Sort by updated_at (newest first)
            todo_lists.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
            
            return {
                "todo_lists": todo_lists,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error getting all todo lists: {str(e)}")
            return {"error": f"حدث خطأ أثناء الحصول على جميع قوائم المهام: {str(e)}"}
    
    def delete_todo_list(self, todo_id: str) -> Dict[str, Any]:
        """
        Delete a todo list
        
        Args:
            todo_id: ID of the todo list
            
        Returns:
            Dictionary containing the result of the operation
        """
        try:
            # Check if the todo list exists
            todo_file = os.path.join(self.todo_dir, f"{todo_id}.json")
            if not os.path.exists(todo_file):
                return {"error": f"قائمة المهام غير موجودة: {todo_id}"}
            
            # Delete the todo list file
            os.remove(todo_file)
            
            return {
                "todo_id": todo_id,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error deleting todo list: {str(e)}")
            return {"error": f"حدث خطأ أثناء حذف قائمة المهام: {str(e)}"}
    
    def export_todo_list(self, todo_id: str, format: str = "markdown") -> Dict[str, Any]:
        """
        Export a todo list to a specific format
        
        Args:
            todo_id: ID of the todo list
            format: Export format (markdown, html, json)
            
        Returns:
            Dictionary containing the exported content
        """
        try:
            # Load the todo list
            todo_list = self._load_todo_list(todo_id)
            if not todo_list:
                return {"error": f"قائمة المهام غير موجودة: {todo_id}"}
            
            if format.lower() == "markdown":
                content = self._export_to_markdown(todo_list)
            elif format.lower() == "html":
                content = self._export_to_html(todo_list)
            elif format.lower() == "json":
                content = json.dumps(todo_list, ensure_ascii=False, indent=2)
            else:
                return {"error": f"تنسيق غير مدعوم: {format}"}
            
            return {
                "content": content,
                "format": format,
                "todo_id": todo_id,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error exporting todo list: {str(e)}")
            return {"error": f"حدث خطأ أثناء تصدير قائمة المهام: {str(e)}"}
    
    def _save_todo_list(self, todo_list: Dict[str, Any]) -> None:
        """
        Save a todo list to file
        
        Args:
            todo_list: Todo list dictionary
        """
        todo_id = todo_list["id"]
        todo_file = os.path.join(self.todo_dir, f"{todo_id}.json")
        
        with open(todo_file, 'w', encoding='utf-8') as f:
            json.dump(todo_list, f, ensure_ascii=False, indent=2)
    
    def _load_todo_list(self, todo_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a todo list from file
        
        Args:
            todo_id: ID of the todo list
            
        Returns:
            Todo list dictionary or None if not found
        """
        todo_file = os.path.join(self.todo_dir, f"{todo_id}.json")
        
        if not os.path.exists(todo_file):
            return None
        
        with open(todo_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _export_to_markdown(self, todo_list: Dict[str, Any]) -> str:
        """
        Export a todo list to Markdown format
        
        Args:
            todo_list: Todo list dictionary
            
        Returns:
            Markdown formatted string
        """
        markdown = f"# {todo_list['title']}\n\n"
        
        if todo_list['description']:
            markdown += f"{todo_list['description']}\n\n"
        
        markdown += f"**المشروع:** {todo_list['project']}\n"
        markdown += f"**تاريخ الإنشاء:** {self._format_timestamp(todo_list['created_at'])}\n"
        markdown += f"**آخر تحديث:** {self._format_timestamp(todo_list['updated_at'])}\n\n"
        
        markdown += "## المهام\n\n"
        
        for task in todo_list['tasks']:
            checkbox = "- [x]" if task['completed'] else "- [ ]"
            priority_marker = ""
            
            if task['priority'] == "high":
                priority_marker = "🔴"
            elif task['priority'] == "medium":
                priority_marker = "🟡"
            elif task['priority'] == "low":
                priority_marker = "🟢"
            
            markdown += f"{checkbox} {priority_marker} {task['task']}"
            
            if task['due_date']:
                markdown += f" (تاريخ الاستحقاق: {self._format_timestamp(task['due_date'])})"
            
            if task['completed'] and task['completed_at']:
                markdown += f" ✓ {self._format_timestamp(task['completed_at'])}"
            
            markdown += "\n"
        
        return markdown
    
    def _export_to_html(self, todo_list: Dict[str, Any]) -> str:
     
(Content truncated due to size limit. Use line ranges to read in chunks)