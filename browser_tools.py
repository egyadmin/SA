"""
Browser Integration Module for Manus Clone

This module handles browser integration including:
- Web browsing and navigation
- Search functionality
- Content extraction
- Screenshot capture
- Form filling and automation
"""

import os
import json
import time
import logging
import tempfile
import base64
from typing import Dict, List, Any, Optional, Union

# Import specialized libraries for browser operations
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.common.action_chains import ActionChains
    import requests
    from bs4 import BeautifulSoup
    import urllib.parse
except ImportError as e:
    logging.error(f"Error importing browser libraries: {str(e)}")
    logging.warning("Browser integration capabilities may be limited")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BrowserTools:
    """Browser Tools class for browser integration"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Browser Tools with configuration
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.screenshot_dir = config.get('SCREENSHOT_FOLDER', os.path.join(os.path.expanduser('~'), 'manus_clone', 'screenshots'))
        self.download_dir = config.get('DOWNLOAD_FOLDER', os.path.join(os.path.expanduser('~'), 'manus_clone', 'downloads'))
        self.browser = None
        self.current_url = None
        self.current_page_source = None
        self.current_page_title = None
        
        # Ensure directories exist
        os.makedirs(self.screenshot_dir, exist_ok=True)
        os.makedirs(self.download_dir, exist_ok=True)
        
        logger.info("Browser Tools initialized successfully")
    
    def get_available_tools(self) -> List[str]:
        """
        Get a list of all available browser tools
        
        Returns:
            List of available tool names
        """
        return [
            "open_url",
            "search_web",
            "extract_content",
            "take_screenshot",
            "fill_form",
            "click_element",
            "scroll_page",
            "download_file",
            "close_browser"
        ]
    
    def execute_tool(self, instruction: str) -> Dict[str, Any]:
        """
        Execute a browser tool based on the instruction
        
        Args:
            instruction: Instruction describing what to do
            
        Returns:
            Dictionary containing the result of the tool execution
        """
        # Parse the instruction to determine which tool to use
        if "فتح رابط" in instruction or "open url" in instruction.lower():
            url = self._extract_url(instruction)
            if url:
                return self.open_url(url)
            else:
                return {"error": "لم يتم تحديد عنوان URL"}
        
        elif "بحث في الويب" in instruction or "search web" in instruction.lower():
            query = self._extract_search_query(instruction)
            if query:
                return self.search_web(query)
            else:
                return {"error": "لم يتم تحديد استعلام البحث"}
        
        elif "استخراج محتوى" in instruction or "extract content" in instruction.lower():
            return self.extract_content()
        
        elif "التقاط لقطة شاشة" in instruction or "take screenshot" in instruction.lower():
            output_path = self._extract_output_path(instruction)
            return self.take_screenshot(output_path)
        
        elif "ملء نموذج" in instruction or "fill form" in instruction.lower():
            form_data = self._extract_form_data(instruction)
            if form_data:
                return self.fill_form(form_data)
            else:
                return {"error": "لم يتم تحديد بيانات النموذج"}
        
        elif "النقر على عنصر" in instruction or "click element" in instruction.lower():
            selector = self._extract_selector(instruction)
            if selector:
                return self.click_element(selector)
            else:
                return {"error": "لم يتم تحديد محدد العنصر"}
        
        elif "تمرير الصفحة" in instruction or "scroll page" in instruction.lower():
            direction = self._extract_scroll_direction(instruction)
            return self.scroll_page(direction)
        
        elif "تنزيل ملف" in instruction or "download file" in instruction.lower():
            url = self._extract_url(instruction)
            output_path = self._extract_output_path(instruction)
            if url:
                return self.download_file(url, output_path)
            else:
                return {"error": "لم يتم تحديد عنوان URL"}
        
        elif "إغلاق المتصفح" in instruction or "close browser" in instruction.lower():
            return self.close_browser()
        
        else:
            return {"error": "لم يتم التعرف على الأداة المطلوبة"}
    
    def open_url(self, url: str) -> Dict[str, Any]:
        """
        Open a URL in the browser
        
        Args:
            url: URL to open
            
        Returns:
            Dictionary containing the result of the operation
        """
        try:
            # Initialize browser if not already initialized
            if not self.browser:
                self._initialize_browser()
            
            # Add http:// if not present
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            # Navigate to the URL
            self.browser.get(url)
            
            # Wait for page to load
            time.sleep(2)
            
            # Update current state
            self.current_url = self.browser.current_url
            self.current_page_source = self.browser.page_source
            self.current_page_title = self.browser.title
            
            # Extract basic page information
            page_info = self._extract_page_info()
            
            return {
                "url": self.current_url,
                "title": self.current_page_title,
                "page_info": page_info,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error opening URL: {str(e)}")
            return {"error": f"حدث خطأ أثناء فتح الرابط: {str(e)}"}
    
    def search_web(self, query: str) -> Dict[str, Any]:
        """
        Search the web for the given query
        
        Args:
            query: Search query
            
        Returns:
            Dictionary containing search results
        """
        try:
            # Initialize browser if not already initialized
            if not self.browser:
                self._initialize_browser()
            
            # Encode the query for URL
            encoded_query = urllib.parse.quote(query)
            
            # Navigate to search engine
            search_url = f"https://www.google.com/search?q={encoded_query}"
            self.browser.get(search_url)
            
            # Wait for results to load
            WebDriverWait(self.browser, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.g"))
            )
            
            # Update current state
            self.current_url = self.browser.current_url
            self.current_page_source = self.browser.page_source
            self.current_page_title = self.browser.title
            
            # Extract search results
            search_results = []
            result_elements = self.browser.find_elements(By.CSS_SELECTOR, "div.g")
            
            for element in result_elements[:10]:  # Limit to first 10 results
                try:
                    title_element = element.find_element(By.CSS_SELECTOR, "h3")
                    title = title_element.text
                    
                    link_element = element.find_element(By.CSS_SELECTOR, "a")
                    link = link_element.get_attribute("href")
                    
                    snippet_element = element.find_element(By.CSS_SELECTOR, "div.VwiC3b")
                    snippet = snippet_element.text
                    
                    search_results.append({
                        "title": title,
                        "link": link,
                        "snippet": snippet
                    })
                except Exception as e:
                    logger.warning(f"Error extracting search result: {str(e)}")
                    continue
            
            return {
                "query": query,
                "results": search_results,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error searching web: {str(e)}")
            return {"error": f"حدث خطأ أثناء البحث في الويب: {str(e)}"}
    
    def extract_content(self) -> Dict[str, Any]:
        """
        Extract content from the current page
        
        Returns:
            Dictionary containing the extracted content
        """
        try:
            if not self.browser:
                return {"error": "المتصفح غير مهيأ"}
            
            # Get page source
            page_source = self.browser.page_source
            
            # Parse the HTML content
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Extract title
            title = soup.title.text.strip() if soup.title else ""
            
            # Extract main content
            main_content = ""
            
            # Try to find main content containers
            main_elements = soup.find_all(['article', 'main', 'div'], class_=lambda c: c and ('content' in c.lower() or 'article' in c.lower()))
            
            if main_elements:
                # Use the first main element found
                main_content = main_elements[0].get_text(separator='\n', strip=True)
            else:
                # Fallback to extracting paragraphs
                paragraphs = soup.find_all('p')
                main_content = '\n'.join([p.get_text(strip=True) for p in paragraphs])
            
            # Extract links
            links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                text = link.get_text(strip=True)
                
                # Convert relative URLs to absolute URLs
                if href.startswith('/'):
                    href = urllib.parse.urljoin(self.current_url, href)
                
                links.append({
                    "href": href,
                    "text": text
                })
            
            # Extract images
            images = []
            for img in soup.find_all('img', src=True):
                src = img['src']
                alt = img.get('alt', '')
                
                # Convert relative URLs to absolute URLs
                if src.startswith('/'):
                    src = urllib.parse.urljoin(self.current_url, src)
                
                images.append({
                    "src": src,
                    "alt": alt
                })
            
            # Extract headings
            headings = []
            for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                headings.append({
                    "level": int(heading.name[1]),
                    "text": heading.get_text(strip=True)
                })
            
            return {
                "url": self.current_url,
                "title": title,
                "content": main_content,
                "links": links,
                "images": images,
                "headings": headings,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error extracting content: {str(e)}")
            return {"error": f"حدث خطأ أثناء استخراج المحتوى: {str(e)}"}
    
    def take_screenshot(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Take a screenshot of the current page
        
        Args:
            output_path: Path where the screenshot should be saved
            
        Returns:
            Dictionary containing the result of the operation
        """
        try:
            if not self.browser:
                return {"error": "المتصفح غير مهيأ"}
            
            # Generate output path if not provided
            if not output_path:
                timestamp = int(time.time())
                filename = f"screenshot_{timestamp}.png"
                output_path = os.path.join(self.screenshot_dir, filename)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Take screenshot
            self.browser.save_screenshot(output_path)
            
            # Get screenshot as base64 for preview
            screenshot_base64 = self.browser.get_screenshot_as_base64()
            
            return {
                "file_path": output_path,
                "file_size": os.path.getsize(output_path),
                "preview": f"data:image/png;base64,{screenshot_base64}",
                "success": True
            }
        except Exception as e:
            logger.error(f"Error taking screenshot: {str(e)}")
            return {"error": f"حدث خطأ أثناء التقاط لقطة الشاشة: {str(e)}"}
    
    def fill_form(self, form_data: Dict[str, str]) -> Dict[str, Any]:
        """
        Fill a form on the current page
        
        Args:
            form_data: Dictionary containing form field names and values
            
        Returns:
            Dictionary containing the result of the operation
        """
        try:
            if not self.browser:
                return {"error": "المتصفح غير مهيأ"}
            
            # Fill form fields
            filled_fields = []
            for field_name, field_value in form_data.items():
                try:
                    # Try to find the field by name, id, or placeholder
                    field = None
                    
                    # Try by name
                    try:
                        field = self.browser.find_element(By.NAME, field_name)
                    except:
                        pass
                    
                    # Try by id if not found by name
                    if not field:
                        try:
                            field = self.browser.find_element(By.ID, field_name)
                        except:
                            pass
                    
                    # Try by placeholder if not found by id
                    if not field:
                        try:
                            field = self.browser.find_element(By.XPATH, f"//input[@placeholder='{field_name}']")
                        except:
                            pass
                    
                    # If field is found, fill it
                    if field:
                        field.clear()
                        field.send_keys(field_value)
                        filled_fields.append(field_name)
                    else:
                        logger.warning(f"Could not find form field: {field_name}")
                except Exception as e:
                    logger.warning(f"Error filling form field {field_name}: {str(e)}")
            
            return {
        
(Content truncated due to size limit. Use line ranges to read in chunks)