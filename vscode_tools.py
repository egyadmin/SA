"""
VS Code Integration Module for Manus Clone

This module handles VS Code integration including:
- Opening files in VS Code
- Creating new files and projects
- Running code from within the application
- Managing workspaces
"""

import os
import json
import time
import logging
import subprocess
import tempfile
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VSCodeTools:
    """VS Code Tools class for VS Code integration"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the VS Code Tools with configuration
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.workspace_dir = config.get('WORKSPACE_FOLDER', os.path.join(os.path.expanduser('~'), 'manus_clone', 'workspaces'))
        self.vscode_path = self._find_vscode_path()
        
        # Ensure workspace directory exists
        os.makedirs(self.workspace_dir, exist_ok=True)
        
        logger.info("VS Code Tools initialized successfully")
        if self.vscode_path:
            logger.info(f"VS Code found at: {self.vscode_path}")
        else:
            logger.warning("VS Code not found. Some features may be limited.")
    
    def get_available_tools(self) -> List[str]:
        """
        Get a list of all available VS Code tools
        
        Returns:
            List of available tool names
        """
        return [
            "open_file",
            "create_file",
            "create_project",
            "run_code",
            "open_workspace"
        ]
    
    def execute_tool(self, instruction: str) -> Dict[str, Any]:
        """
        Execute a VS Code tool based on the instruction
        
        Args:
            instruction: Instruction describing what to do
            
        Returns:
            Dictionary containing the result of the tool execution
        """
        # Parse the instruction to determine which tool to use
        if "فتح ملف" in instruction or "open file" in instruction.lower():
            file_path = self._extract_file_path(instruction)
            if file_path:
                return self.open_file(file_path)
            else:
                return {"error": "لم يتم تحديد مسار الملف"}
        
        elif "إنشاء ملف" in instruction or "create file" in instruction.lower():
            file_path = self._extract_file_path(instruction)
            content = self._extract_content(instruction)
            if file_path:
                return self.create_file(file_path, content)
            else:
                return {"error": "لم يتم تحديد مسار الملف"}
        
        elif "إنشاء مشروع" in instruction or "create project" in instruction.lower():
            project_name = self._extract_project_name(instruction)
            project_type = self._extract_project_type(instruction)
            if project_name:
                return self.create_project(project_name, project_type)
            else:
                return {"error": "لم يتم تحديد اسم المشروع"}
        
        elif "تشغيل كود" in instruction or "run code" in instruction.lower():
            code = self._extract_code(instruction)
            language = self._extract_language(instruction)
            if code:
                return self.run_code(code, language)
            else:
                return {"error": "لم يتم تحديد الكود"}
        
        elif "فتح مساحة عمل" in instruction or "open workspace" in instruction.lower():
            workspace_name = self._extract_workspace_name(instruction)
            if workspace_name:
                return self.open_workspace(workspace_name)
            else:
                return {"error": "لم يتم تحديد اسم مساحة العمل"}
        
        else:
            return {"error": "لم يتم التعرف على الأداة المطلوبة"}
    
    def open_file(self, file_path: str) -> Dict[str, Any]:
        """
        Open a file in VS Code
        
        Args:
            file_path: Path to the file to open
            
        Returns:
            Dictionary containing the result of the operation
        """
        try:
            if not os.path.exists(file_path):
                return {"error": f"الملف غير موجود: {file_path}"}
            
            if not self.vscode_path:
                return {"error": "لم يتم العثور على VS Code"}
            
            # Open the file in VS Code
            subprocess.Popen([self.vscode_path, file_path])
            
            return {
                "file_path": file_path,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error opening file in VS Code: {str(e)}")
            return {"error": f"حدث خطأ أثناء فتح الملف في VS Code: {str(e)}"}
    
    def create_file(self, file_path: str, content: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new file and optionally open it in VS Code
        
        Args:
            file_path: Path where the file should be created
            content: Content to write to the file
            
        Returns:
            Dictionary containing the result of the operation
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # Create the file
            with open(file_path, 'w', encoding='utf-8') as f:
                if content:
                    f.write(content)
            
            # Open the file in VS Code if available
            if self.vscode_path:
                subprocess.Popen([self.vscode_path, file_path])
            
            return {
                "file_path": file_path,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error creating file: {str(e)}")
            return {"error": f"حدث خطأ أثناء إنشاء الملف: {str(e)}"}
    
    def create_project(self, project_name: str, project_type: str = "python") -> Dict[str, Any]:
        """
        Create a new project with template files
        
        Args:
            project_name: Name of the project
            project_type: Type of project (python, web, node, etc.)
            
        Returns:
            Dictionary containing the result of the operation
        """
        try:
            # Create project directory
            project_dir = os.path.join(self.workspace_dir, project_name)
            os.makedirs(project_dir, exist_ok=True)
            
            # Create template files based on project type
            if project_type.lower() == "python":
                self._create_python_project(project_dir, project_name)
            elif project_type.lower() == "web":
                self._create_web_project(project_dir, project_name)
            elif project_type.lower() == "node":
                self._create_node_project(project_dir, project_name)
            else:
                # Default to empty project with README
                readme_path = os.path.join(project_dir, "README.md")
                with open(readme_path, 'w', encoding='utf-8') as f:
                    f.write(f"# {project_name}\n\nA new project created with Manus Clone.")
            
            # Open the project in VS Code if available
            if self.vscode_path:
                subprocess.Popen([self.vscode_path, project_dir])
            
            return {
                "project_dir": project_dir,
                "project_name": project_name,
                "project_type": project_type,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error creating project: {str(e)}")
            return {"error": f"حدث خطأ أثناء إنشاء المشروع: {str(e)}"}
    
    def run_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """
        Run code and return the result
        
        Args:
            code: Code to run
            language: Programming language
            
        Returns:
            Dictionary containing the result of the operation
        """
        try:
            # Create a temporary file for the code
            with tempfile.NamedTemporaryFile(suffix=self._get_file_extension(language), delete=False) as temp_file:
                temp_file_path = temp_file.name
                temp_file.write(code.encode('utf-8'))
            
            # Run the code based on the language
            if language.lower() == "python":
                result = self._run_python_code(temp_file_path)
            elif language.lower() == "javascript":
                result = self._run_javascript_code(temp_file_path)
            elif language.lower() == "bash":
                result = self._run_bash_code(temp_file_path)
            else:
                os.unlink(temp_file_path)
                return {"error": f"لغة البرمجة غير مدعومة: {language}"}
            
            # Clean up the temporary file
            os.unlink(temp_file_path)
            
            return {
                "output": result,
                "language": language,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error running code: {str(e)}")
            return {"error": f"حدث خطأ أثناء تشغيل الكود: {str(e)}"}
    
    def open_workspace(self, workspace_name: str) -> Dict[str, Any]:
        """
        Open a workspace in VS Code
        
        Args:
            workspace_name: Name of the workspace
            
        Returns:
            Dictionary containing the result of the operation
        """
        try:
            # Check if the workspace exists
            workspace_dir = os.path.join(self.workspace_dir, workspace_name)
            if not os.path.exists(workspace_dir):
                return {"error": f"مساحة العمل غير موجودة: {workspace_name}"}
            
            if not self.vscode_path:
                return {"error": "لم يتم العثور على VS Code"}
            
            # Open the workspace in VS Code
            subprocess.Popen([self.vscode_path, workspace_dir])
            
            return {
                "workspace_dir": workspace_dir,
                "workspace_name": workspace_name,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error opening workspace: {str(e)}")
            return {"error": f"حدث خطأ أثناء فتح مساحة العمل: {str(e)}"}
    
    def _find_vscode_path(self) -> Optional[str]:
        """
        Find the path to VS Code executable
        
        Returns:
            Path to VS Code executable or None if not found
        """
        # Common paths for VS Code
        common_paths = [
            # Windows
            r"C:\Program Files\Microsoft VS Code\Code.exe",
            r"C:\Program Files (x86)\Microsoft VS Code\Code.exe",
            # macOS
            "/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code",
            # Linux
            "/usr/bin/code",
            "/usr/local/bin/code",
            "/snap/bin/code"
        ]
        
        # Check common paths
        for path in common_paths:
            if os.path.exists(path):
                return path
        
        # Try to find using 'which' command on Unix-like systems
        try:
            result = subprocess.run(["which", "code"], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        
        # Try to find using 'where' command on Windows
        try:
            result = subprocess.run(["where", "code"], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip().split('\n')[0]
        except:
            pass
        
        return None
    
    def _create_python_project(self, project_dir: str, project_name: str) -> None:
        """
        Create a Python project with template files
        
        Args:
            project_dir: Project directory
            project_name: Name of the project
        """
        # Create main.py
        main_path = os.path.join(project_dir, "main.py")
        with open(main_path, 'w', encoding='utf-8') as f:
            f.write(f"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def main():
    print("Welcome to {project_name}!")
    
if __name__ == "__main__":
    main()
""")
        
        # Create README.md
        readme_path = os.path.join(project_dir, "README.md")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"""# {project_name}

A Python project created with Manus Clone.

## Getting Started

1. Run the main script:
   ```
   python main.py
   ```

## Project Structure

- `main.py`: Main entry point
- `README.md`: This file
""")
        
        # Create requirements.txt
        req_path = os.path.join(project_dir, "requirements.txt")
        with open(req_path, 'w', encoding='utf-8') as f:
            f.write("# Add your dependencies here\n")
        
        # Create .gitignore
        gitignore_path = os.path.join(project_dir, ".gitignore")
        with open(gitignore_path, 'w', encoding='utf-8') as f:
            f.write("""# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Virtual environments
venv/
env/
.env/

# Distribution / packaging
dist/
build/
*.egg-info/

# IDE files
.idea/
.vscode/
*.swp
*.swo
""")
    
    def _create_web_project(self, project_dir: str, project_name: str) -> None:
        """
        Create a web project with template files
        
        Args:
            project_dir: Project directory
            project_name: Name of the project
        """
        # Create index.html
        index_path = os.path.join(project_dir, "index.html")
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{project_name}</title>
    <link rel="stylesheet" href="css/style.css">
</head>
<body>
    <header>
        <h1>{project_name}</h1>
    </header>
    
    <main>
        <p>Welcome to {project_name}!</p>
    </main>
    
    <footer>
        <p>&copy; {time.strftime('%Y')} {project_name}</p>
    </footer>
    
    <script src="js/main.js"></script>
</body>
</html>
""")
        
        # Create CSS directory and style.css
        css_dir = os.path.join(project_dir, "css")
        os.makedirs(css_dir, exist_ok=True)
        css_path = os.path.join(css_dir, "style.css")
        with open(css_path, 'w', encoding='utf-8') as f:
            f.write("""/* Reset and base styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: Arial, sans-serif;
    line-height: 1.6;
    color: #333;
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

header {
    text-align: center;
    margin-bottom: 30px;
}

main {
    min-height: 70vh;
}

footer {
    text-align: center;
    margin-top: 30px;
    padding-top: 10px;
    border-top: 1px solid #eee;
}

/* Responsive design */
@media (max-width: 768px) {
    body {
        padding: 10px;
    }
}
""")
        
        # Create JS directory and main.js
        js_dir = os.path.join(project_dir, "js")
        os.makedirs(js_dir, exist_ok=True)
        js_path = os.path.join(js_dir, "main.js")
        with open(js_path, 'w', encoding='utf-8') as f:
            f.write("""// Main JavaScript file

document.addEventListener('DOMContentLoaded', function() {
    console.log('Document loaded and ready!');
});
""")
        
        # Create README.md
        readme_path = os.path.join(project_dir, "README.md")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.wri
(Content truncated due to size limit. Use line ranges to read in chunks)