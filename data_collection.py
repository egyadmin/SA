"""
Data Collection Module for Manus Clone

This module handles data collection and organization including:
- Collecting data from various sources
- Organizing data into tables
- Tracking data sources
- Exporting data in various formats
"""

import os
import json
import time
import logging
import csv
import pandas as pd
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataCollection:
    """Data Collection class for collecting and organizing data"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Data Collection with configuration
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.data_dir = config.get('DATA_FOLDER', os.path.join(os.path.expanduser('~'), 'manus_clone', 'data'))
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        logger.info("Data Collection initialized successfully")
    
    def create_data_table(self, name: str, description: str = "", columns: List[str] = None) -> Dict[str, Any]:
        """
        Create a new data table
        
        Args:
            name: Name of the data table
            description: Description of the data table
            columns: List of column names
            
        Returns:
            Dictionary containing the created data table information
        """
        try:
            # Generate a unique ID for the data table
            table_id = f"table_{int(time.time())}"
            
            # Create the data table structure
            data_table = {
                "id": table_id,
                "name": name,
                "description": description,
                "created_at": time.time(),
                "updated_at": time.time(),
                "columns": columns or [],
                "rows": [],
                "sources": []
            }
            
            # Save the data table
            self._save_data_table(data_table)
            
            return {
                "data_table": data_table,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error creating data table: {str(e)}")
            return {"error": f"حدث خطأ أثناء إنشاء جدول البيانات: {str(e)}"}
    
    def add_row(self, table_id: str, row_data: Dict[str, Any], source: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Add a row to a data table
        
        Args:
            table_id: ID of the data table
            row_data: Dictionary containing the row data
            source: Source information for the data
            
        Returns:
            Dictionary containing the updated data table
        """
        try:
            # Load the data table
            data_table = self._load_data_table(table_id)
            if not data_table:
                return {"error": f"جدول البيانات غير موجود: {table_id}"}
            
            # Generate a unique ID for the row
            row_id = f"row_{int(time.time())}_{len(data_table['rows'])}"
            
            # Create the row structure
            row_item = {
                "id": row_id,
                "data": row_data,
                "created_at": time.time(),
                "source_id": None
            }
            
            # Add source information if provided
            if source:
                source_id = self._add_source(data_table, source)
                row_item["source_id"] = source_id
            
            # Add the row to the data table
            data_table["rows"].append(row_item)
            data_table["updated_at"] = time.time()
            
            # Update columns if new keys are found
            for key in row_data.keys():
                if key not in data_table["columns"]:
                    data_table["columns"].append(key)
            
            # Save the updated data table
            self._save_data_table(data_table)
            
            return {
                "data_table": data_table,
                "row": row_item,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error adding row: {str(e)}")
            return {"error": f"حدث خطأ أثناء إضافة صف: {str(e)}"}
    
    def update_row(self, table_id: str, row_id: str, row_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a row in a data table
        
        Args:
            table_id: ID of the data table
            row_id: ID of the row
            row_data: Dictionary containing the updated row data
            
        Returns:
            Dictionary containing the updated data table
        """
        try:
            # Load the data table
            data_table = self._load_data_table(table_id)
            if not data_table:
                return {"error": f"جدول البيانات غير موجود: {table_id}"}
            
            # Find the row
            row_found = False
            for row in data_table["rows"]:
                if row["id"] == row_id:
                    row["data"] = row_data
                    row_found = True
                    break
            
            if not row_found:
                return {"error": f"الصف غير موجود: {row_id}"}
            
            data_table["updated_at"] = time.time()
            
            # Update columns if new keys are found
            for key in row_data.keys():
                if key not in data_table["columns"]:
                    data_table["columns"].append(key)
            
            # Save the updated data table
            self._save_data_table(data_table)
            
            return {
                "data_table": data_table,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error updating row: {str(e)}")
            return {"error": f"حدث خطأ أثناء تحديث الصف: {str(e)}"}
    
    def delete_row(self, table_id: str, row_id: str) -> Dict[str, Any]:
        """
        Delete a row from a data table
        
        Args:
            table_id: ID of the data table
            row_id: ID of the row
            
        Returns:
            Dictionary containing the updated data table
        """
        try:
            # Load the data table
            data_table = self._load_data_table(table_id)
            if not data_table:
                return {"error": f"جدول البيانات غير موجود: {table_id}"}
            
            # Find and remove the row
            row_found = False
            for i, row in enumerate(data_table["rows"]):
                if row["id"] == row_id:
                    data_table["rows"].pop(i)
                    row_found = True
                    break
            
            if not row_found:
                return {"error": f"الصف غير موجود: {row_id}"}
            
            data_table["updated_at"] = time.time()
            
            # Save the updated data table
            self._save_data_table(data_table)
            
            return {
                "data_table": data_table,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error deleting row: {str(e)}")
            return {"error": f"حدث خطأ أثناء حذف الصف: {str(e)}"}
    
    def get_data_table(self, table_id: str) -> Dict[str, Any]:
        """
        Get a data table by ID
        
        Args:
            table_id: ID of the data table
            
        Returns:
            Dictionary containing the data table
        """
        try:
            # Load the data table
            data_table = self._load_data_table(table_id)
            if not data_table:
                return {"error": f"جدول البيانات غير موجود: {table_id}"}
            
            return {
                "data_table": data_table,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error getting data table: {str(e)}")
            return {"error": f"حدث خطأ أثناء الحصول على جدول البيانات: {str(e)}"}
    
    def get_all_data_tables(self) -> Dict[str, Any]:
        """
        Get all data tables
        
        Returns:
            Dictionary containing the data tables
        """
        try:
            data_tables = []
            
            # Get all data table files
            for filename in os.listdir(self.data_dir):
                if filename.endswith('.json') and filename.startswith('table_'):
                    table_id = filename[:-5]  # Remove .json extension
                    data_table = self._load_data_table(table_id)
                    
                    if data_table:
                        data_tables.append(data_table)
            
            # Sort by updated_at (newest first)
            data_tables.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
            
            return {
                "data_tables": data_tables,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error getting all data tables: {str(e)}")
            return {"error": f"حدث خطأ أثناء الحصول على جميع جداول البيانات: {str(e)}"}
    
    def delete_data_table(self, table_id: str) -> Dict[str, Any]:
        """
        Delete a data table
        
        Args:
            table_id: ID of the data table
            
        Returns:
            Dictionary containing the result of the operation
        """
        try:
            # Check if the data table exists
            table_file = os.path.join(self.data_dir, f"{table_id}.json")
            if not os.path.exists(table_file):
                return {"error": f"جدول البيانات غير موجود: {table_id}"}
            
            # Delete the data table file
            os.remove(table_file)
            
            return {
                "table_id": table_id,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error deleting data table: {str(e)}")
            return {"error": f"حدث خطأ أثناء حذف جدول البيانات: {str(e)}"}
    
    def export_data_table(self, table_id: str, format: str = "csv", output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Export a data table to a specific format
        
        Args:
            table_id: ID of the data table
            format: Export format (csv, excel, json, html, markdown)
            output_path: Path where the exported file should be saved
            
        Returns:
            Dictionary containing the result of the operation
        """
        try:
            # Load the data table
            data_table = self._load_data_table(table_id)
            if not data_table:
                return {"error": f"جدول البيانات غير موجود: {table_id}"}
            
            # Convert to DataFrame for easier export
            df = self._convert_to_dataframe(data_table)
            
            # Generate output path if not provided
            if not output_path:
                filename = f"{data_table['name']}_{int(time.time())}"
                if format.lower() == "csv":
                    filename += ".csv"
                elif format.lower() == "excel":
                    filename += ".xlsx"
                elif format.lower() == "json":
                    filename += ".json"
                elif format.lower() == "html":
                    filename += ".html"
                elif format.lower() == "markdown":
                    filename += ".md"
                else:
                    return {"error": f"تنسيق غير مدعوم: {format}"}
                
                output_path = os.path.join(self.data_dir, filename)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Export to the specified format
            if format.lower() == "csv":
                df.to_csv(output_path, index=False, encoding='utf-8')
            elif format.lower() == "excel":
                df.to_excel(output_path, index=False)
            elif format.lower() == "json":
                df.to_json(output_path, orient='records', force_ascii=False, indent=2)
            elif format.lower() == "html":
                df.to_html(output_path, index=False, encoding='utf-8')
            elif format.lower() == "markdown":
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"# {data_table['name']}\n\n")
                    if data_table['description']:
                        f.write(f"{data_table['description']}\n\n")
                    f.write(df.to_markdown(index=False))
                    f.write("\n\n## المصادر\n\n")
                    for source in data_table['sources']:
                        f.write(f"- {source['description']}")
                        if source.get('url'):
                            f.write(f": [{source['url']}]({source['url']})")
                        f.write("\n")
            else:
                return {"error": f"تنسيق غير مدعوم: {format}"}
            
            return {
                "file_path": output_path,
                "format": format,
                "table_id": table_id,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error exporting data table: {str(e)}")
            return {"error": f"حدث خطأ أثناء تصدير جدول البيانات: {str(e)}"}
    
    def import_data(self, file_path: str, name: Optional[str] = None, description: str = "") -> Dict[str, Any]:
        """
        Import data from a file into a new data table
        
        Args:
            file_path: Path to the file to import
            name: Name for the new data table
            description: Description for the new data table
            
        Returns:
            Dictionary containing the created data table
        """
        try:
            # Determine file type from extension
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Read the data based on file type
            if file_ext == '.csv':
                df = pd.read_csv(file_path)
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif file_ext == '.json':
                df = pd.read_json(file_path)
            else:
                return {"error": f"نوع الملف غير مدعوم: {file_ext}"}
            
            # Generate name if not provided
            if not name:
                name = os.path.basename(file_path)
            
            # Create a new data table
            result = self.create_data_table(name, description, list(df.columns))
            
            if "error" in result:
                return result
            
            data_table = result["data_table"]
            table_id = data_table["id"]
            
            # Add source information
            source = {
                "description": f"Imported from {os.path.basename(file_path)}",
                "type": "file",
                "path": file_path
            }
            
            # Add rows from the DataFrame
            for _, row in df.iterrows():
                row_data = row.to_dict()
                self.add_row(table_id, row_data, source)
            
            # Get the updated data table
            return self.get_data_table(table_id)
        except Exception as e:
            logger.error(f"Error importing data: {str(e)}")
            return {"error": f"حدث خطأ أثناء استيراد البيانات: {str(e)}"}
    
    def add_source(self, table_id: str, source: Dict[str, str]) -> Dict[str, Any]:
        """
        Add a source to a data table
        
        Args:
            table_id: ID of the data t
(Content truncated due to size limit. Use line ranges to read in chunks)