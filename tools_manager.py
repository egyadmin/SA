"""
Tools Manager Module for Manus Clone

This module manages all the tools available to the AI agent and handles tool execution.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union

# Import tool modules
from tools.file_tools.file_tools import FileTools
from tools.web_tools.web_tools import WebTools
from tools.data_tools.data_tools import DataTools
from tools.audio_video_tools.audio_video_tools import AudioVideoTools
from tools.desktop_tools.desktop_tools import DesktopTools
from tools.language_tools.language_tools import LanguageTools

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ToolsManager:
    """Tools Manager class for coordinating all available tools"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Tools Manager with configuration
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        
        # Initialize tool modules
        self.file_tools = FileTools(config)
        self.web_tools = WebTools(config)
        self.data_tools = DataTools(config)
        self.audio_video_tools = AudioVideoTools(config)
        self.desktop_tools = DesktopTools(config)
        self.language_tools = LanguageTools(config)
        
        logger.info("Tools Manager initialized successfully")
    
    def get_available_tools(self) -> Dict[str, List[str]]:
        """
        Get a list of all available tools
        
        Returns:
            Dictionary of tool categories and their available tools
        """
        available_tools = {
            "file_tools": self.file_tools.get_available_tools(),
            "web_tools": self.web_tools.get_available_tools(),
            "data_tools": self.data_tools.get_available_tools(),
            "audio_video_tools": self.audio_video_tools.get_available_tools(),
            "desktop_tools": self.desktop_tools.get_available_tools(),
            "language_tools": self.language_tools.get_available_tools()
        }
        
        return available_tools
    
    def execute_file_tool(self, instruction: str) -> Dict[str, Any]:
        """
        Execute a file-related tool based on the instruction
        
        Args:
            instruction: Instruction describing what to do
            
        Returns:
            Dictionary containing the result of the tool execution
        """
        try:
            return self.file_tools.execute_tool(instruction)
        except Exception as e:
            logger.error(f"Error executing file tool: {str(e)}")
            return {"result": f"حدث خطأ أثناء تنفيذ أداة الملفات: {str(e)}"}
    
    def execute_web_tool(self, instruction: str) -> Dict[str, Any]:
        """
        Execute a web-related tool based on the instruction
        
        Args:
            instruction: Instruction describing what to do
            
        Returns:
            Dictionary containing the result of the tool execution
        """
        try:
            return self.web_tools.execute_tool(instruction)
        except Exception as e:
            logger.error(f"Error executing web tool: {str(e)}")
            return {"result": f"حدث خطأ أثناء تنفيذ أداة الويب: {str(e)}"}
    
    def execute_data_tool(self, instruction: str) -> Dict[str, Any]:
        """
        Execute a data-related tool based on the instruction
        
        Args:
            instruction: Instruction describing what to do
            
        Returns:
            Dictionary containing the result of the tool execution
        """
        try:
            return self.data_tools.execute_tool(instruction)
        except Exception as e:
            logger.error(f"Error executing data tool: {str(e)}")
            return {"result": f"حدث خطأ أثناء تنفيذ أداة البيانات: {str(e)}"}
    
    def execute_audio_tool(self, instruction: str) -> Dict[str, Any]:
        """
        Execute an audio-related tool based on the instruction
        
        Args:
            instruction: Instruction describing what to do
            
        Returns:
            Dictionary containing the result of the tool execution
        """
        try:
            return self.audio_video_tools.execute_audio_tool(instruction)
        except Exception as e:
            logger.error(f"Error executing audio tool: {str(e)}")
            return {"result": f"حدث خطأ أثناء تنفيذ أداة الصوت: {str(e)}"}
    
    def execute_video_tool(self, instruction: str) -> Dict[str, Any]:
        """
        Execute a video-related tool based on the instruction
        
        Args:
            instruction: Instruction describing what to do
            
        Returns:
            Dictionary containing the result of the tool execution
        """
        try:
            return self.audio_video_tools.execute_video_tool(instruction)
        except Exception as e:
            logger.error(f"Error executing video tool: {str(e)}")
            return {"result": f"حدث خطأ أثناء تنفيذ أداة الفيديو: {str(e)}"}
    
    def execute_desktop_tool(self, instruction: str) -> Dict[str, Any]:
        """
        Execute a desktop-related tool based on the instruction
        
        Args:
            instruction: Instruction describing what to do
            
        Returns:
            Dictionary containing the result of the tool execution
        """
        try:
            return self.desktop_tools.execute_tool(instruction)
        except Exception as e:
            logger.error(f"Error executing desktop tool: {str(e)}")
            return {"result": f"حدث خطأ أثناء تنفيذ أداة سطح المكتب: {str(e)}"}
    
    def execute_language_tool(self, instruction: str) -> Dict[str, Any]:
        """
        Execute a language-related tool based on the instruction
        
        Args:
            instruction: Instruction describing what to do
            
        Returns:
            Dictionary containing the result of the tool execution
        """
        try:
            return self.language_tools.execute_tool(instruction)
        except Exception as e:
            logger.error(f"Error executing language tool: {str(e)}")
            return {"result": f"حدث خطأ أثناء تنفيذ أداة اللغة: {str(e)}"}
