"""
Core AI Agent Module for Manus Clone

This module implements the core functionality of the AI agent, including:
- Handling user requests
- Managing conversation context
- Coordinating tool usage
- Processing responses
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union

# Import LLM providers
import openai
from anthropic import Anthropic

# Import tools manager
from .tools_manager import ToolsManager
from .memory import Memory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AIAgent:
    """Main AI Agent class that handles user interactions and tool usage"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the AI Agent with configuration
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.tools_manager = ToolsManager(config)
        self.memory = Memory(config)
        self.conversation_id = None
        self.status_updates = []
        self.current_task = None
        
        # Initialize API clients based on config
        self._init_llm_clients()
        
        logger.info("AI Agent initialized successfully")
    
    def _init_llm_clients(self):
        """Initialize language model clients based on configuration"""
        # OpenAI setup
        if self.config.get("use_openai", True):
            openai.api_key = self.config.get("openai_api_key", os.getenv("OPENAI_API_KEY"))
            self.openai_model = self.config.get("openai_model", "gpt-4o")
            logger.info(f"OpenAI initialized with model: {self.openai_model}")
        
        # Anthropic setup
        if self.config.get("use_anthropic", False):
            anthropic_api_key = self.config.get("anthropic_api_key", os.getenv("ANTHROPIC_API_KEY"))
            self.anthropic_client = Anthropic(api_key=anthropic_api_key)
            self.anthropic_model = self.config.get("anthropic_model", "claude-3-opus-20240229")
            logger.info(f"Anthropic initialized with model: {self.anthropic_model}")
    
    def process_request(self, user_input: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a user request and return a response
        
        Args:
            user_input: The user's input text
            conversation_id: Optional conversation ID for continuing a conversation
            
        Returns:
            Dictionary containing the response and any additional information
        """
        # Set or create conversation ID
        if conversation_id:
            self.conversation_id = conversation_id
        elif not self.conversation_id:
            self.conversation_id = f"conv_{int(time.time())}"
        
        # Reset status updates for new request
        self.status_updates = []
        self.current_task = user_input
        
        # Add initial status update
        self._add_status_update("تحليل الطلب...", "analysis")
        
        # Store user message in memory
        self.memory.add_user_message(user_input, self.conversation_id)
        
        # Plan execution steps
        self._add_status_update("تخطيط خطوات التنفيذ...", "planning")
        execution_plan = self._plan_execution(user_input)
        
        # Execute the plan
        self._add_status_update("بدء تنفيذ المهمة...", "execution")
        result = self._execute_plan(execution_plan)
        
        # Store assistant response in memory
        self.memory.add_assistant_message(result["response"], self.conversation_id)
        
        # Add completion status
        self._add_status_update("اكتملت المهمة", "complete")
        
        # Return the complete response
        return {
            "response": result["response"],
            "conversation_id": self.conversation_id,
            "status_updates": self.status_updates,
            "tools_used": result.get("tools_used", []),
            "attachments": result.get("attachments", [])
        }
    
    def _plan_execution(self, user_input: str) -> List[Dict[str, Any]]:
        """
        Plan the execution steps for a user request
        
        Args:
            user_input: The user's input text
            
        Returns:
            List of execution steps
        """
        # Get conversation history
        conversation_history = self.memory.get_conversation_history(self.conversation_id)
        
        # Use LLM to generate execution plan
        messages = [
            {"role": "system", "content": "أنت مساعد ذكي يقوم بتخطيط خطوات تنفيذ طلبات المستخدم. قم بتحليل طلب المستخدم وتحديد الأدوات اللازمة والخطوات المطلوبة للتنفيذ."},
            {"role": "user", "content": f"طلب المستخدم: {user_input}\n\nقم بإنشاء خطة تنفيذ مفصلة تتضمن الخطوات والأدوات اللازمة."}
        ]
        
        # Add conversation history for context if available
        if conversation_history:
            for message in conversation_history[-5:]:  # Use last 5 messages for context
                messages.append({"role": message["role"], "content": message["content"]})
        
        try:
            response = openai.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                temperature=0.2,
                max_tokens=1000
            )
            
            plan_text = response.choices[0].message.content
            
            # Parse the plan text into structured steps
            # This is a simplified implementation - in a real system, you'd want more robust parsing
            steps = []
            for line in plan_text.split('\n'):
                if line.strip() and ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        step_name = parts[0].strip()
                        step_description = parts[1].strip()
                        steps.append({
                            "name": step_name,
                            "description": step_description,
                            "tools": self._extract_tools_from_description(step_description)
                        })
            
            return steps
        
        except Exception as e:
            logger.error(f"Error planning execution: {str(e)}")
            return [{"name": "direct_response", "description": "الرد المباشر على المستخدم", "tools": ["none"]}]
    
    def _extract_tools_from_description(self, description: str) -> List[str]:
        """Extract potential tools from a step description"""
        tools = []
        tool_keywords = {
            "file": ["file", "ملف", "قراءة", "كتابة", "تحميل"],
            "web": ["web", "browser", "متصفح", "بحث", "موقع"],
            "data": ["data", "بيانات", "تحليل", "رسم", "إحصاء"],
            "audio": ["audio", "صوت", "تسجيل", "speech"],
            "video": ["video", "فيديو", "مقطع"],
            "desktop": ["desktop", "screen", "شاشة", "نافذة", "تطبيق"],
            "language": ["translate", "ترجمة", "لغة", "language"]
        }
        
        for tool_type, keywords in tool_keywords.items():
            for keyword in keywords:
                if keyword in description.lower():
                    tools.append(tool_type)
                    break
        
        return tools if tools else ["none"]
    
    def _execute_plan(self, execution_plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute a plan and return the results
        
        Args:
            execution_plan: List of execution steps
            
        Returns:
            Dictionary containing the response and any additional information
        """
        results = []
        tools_used = []
        attachments = []
        
        for i, step in enumerate(execution_plan):
            step_name = step["name"]
            step_description = step["description"]
            
            # Update status
            self._add_status_update(f"تنفيذ: {step_name}", "step_execution")
            
            # Execute the appropriate tools based on the step
            step_result = self._execute_step(step)
            
            # Track tools used
            if "tools_used" in step_result:
                tools_used.extend(step_result["tools_used"])
            
            # Track attachments
            if "attachments" in step_result:
                attachments.extend(step_result["attachments"])
            
            # Add the result to our results list
            results.append({
                "step": step_name,
                "result": step_result.get("result", "تم التنفيذ بنجاح")
            })
            
            # Update status with completion of this step
            self._add_status_update(f"اكتمل: {step_name}", "step_complete")
        
        # Generate final response using LLM
        self._add_status_update("إعداد الرد النهائي...", "response_generation")
        
        response = self._generate_final_response(self.current_task, results)
        
        return {
            "response": response,
            "tools_used": tools_used,
            "attachments": attachments
        }
    
    def _execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single step in the plan
        
        Args:
            step: Dictionary containing step information
            
        Returns:
            Dictionary containing the result of the step execution
        """
        tools_used = []
        attachments = []
        result = ""
        
        # Determine which tools to use based on the step
        for tool_type in step.get("tools", ["none"]):
            if tool_type == "file":
                # Execute file-related tools
                file_result = self.tools_manager.execute_file_tool(step["description"])
                result += file_result.get("result", "")
                tools_used.append("file_tools")
                if "attachments" in file_result:
                    attachments.extend(file_result["attachments"])
            
            elif tool_type == "web":
                # Execute web-related tools
                web_result = self.tools_manager.execute_web_tool(step["description"])
                result += web_result.get("result", "")
                tools_used.append("web_tools")
                if "attachments" in web_result:
                    attachments.extend(web_result["attachments"])
            
            elif tool_type == "data":
                # Execute data-related tools
                data_result = self.tools_manager.execute_data_tool(step["description"])
                result += data_result.get("result", "")
                tools_used.append("data_tools")
                if "attachments" in data_result:
                    attachments.extend(data_result["attachments"])
            
            elif tool_type == "audio":
                # Execute audio-related tools
                audio_result = self.tools_manager.execute_audio_tool(step["description"])
                result += audio_result.get("result", "")
                tools_used.append("audio_video_tools")
                if "attachments" in audio_result:
                    attachments.extend(audio_result["attachments"])
            
            elif tool_type == "video":
                # Execute video-related tools
                video_result = self.tools_manager.execute_video_tool(step["description"])
                result += video_result.get("result", "")
                tools_used.append("audio_video_tools")
                if "attachments" in video_result:
                    attachments.extend(video_result["attachments"])
            
            elif tool_type == "desktop":
                # Execute desktop-related tools
                desktop_result = self.tools_manager.execute_desktop_tool(step["description"])
                result += desktop_result.get("result", "")
                tools_used.append("desktop_tools")
                if "attachments" in desktop_result:
                    attachments.extend(desktop_result["attachments"])
            
            elif tool_type == "language":
                # Execute language-related tools
                language_result = self.tools_manager.execute_language_tool(step["description"])
                result += language_result.get("result", "")
                tools_used.append("language_tools")
                if "attachments" in language_result:
                    attachments.extend(language_result["attachments"])
        
        return {
            "result": result,
            "tools_used": tools_used,
            "attachments": attachments
        }
    
    def _generate_final_response(self, user_input: str, results: List[Dict[str, Any]]) -> str:
        """
        Generate a final response based on the execution results
        
        Args:
            user_input: The original user input
            results: List of results from executing the plan
            
        Returns:
            Final response text
        """
        # Get conversation history
        conversation_history = self.memory.get_conversation_history(self.conversation_id)
        
        # Prepare the prompt for the LLM
        results_text = "\n".join([f"{r['step']}: {r['result']}" for r in results])
        
        messages = [
            {"role": "system", "content": "أنت مساعد ذكي يقوم بإعداد ردود مفصلة ومفيدة بناءً على نتائج تنفيذ المهام. قم بتلخيص النتائج وتقديم إجابة شاملة للمستخدم."},
            {"role": "user", "content": f"طلب المستخدم: {user_input}\n\nنتائج التنفيذ:\n{results_text}\n\nقم بإعداد رد شامل ومفيد للمستخدم بناءً على هذه النتائج."}
        ]
        
        # Add conversation history for context if available
        if conversation_history:
            for message in conversation_history[-5:]:  # Use last 5 messages for context
                messages.append({"role": message["role"], "content": message["content"]})
        
        try:
            response = openai.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Error generating final response: {str(e)}")
            return "عذراً، حدث خطأ أثناء إعداد الرد النهائي. يرجى المحاولة مرة أخرى."
    
    def _add_status_update(self, message: str, status_type: str):
        """
        Add a status update to the current request
        
        Args:
            message: Status message
            status_type: Type of status update
        """
        timestamp = time.time()
        self.status_updates.append({
            "message": message,
            "type": status_type,
            "timestamp": timestamp
        })
        logger.info(f"Status update: {message} ({status_type})")
    
    def get_status_updates(self) -> List[Dict[str, Any]]:
        """
        Get the current status updates
        
        Returns:
            List of status updates
        """
        return self.status_updates
    
    def get_available_tools(self) -> Dict[str, List[str]]:
        """
        Get a list of all available tools
        
        Returns:
            Dictionary of tool categories and their available tools
        """
        return self.tools_manager.get_available_tools()
