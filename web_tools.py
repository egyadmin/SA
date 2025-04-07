"""
Web Tools Module for Manus Clone

This module handles web-related operations including:
- Web search
- Web page content extraction
- Web browsing
- File downloads from the web
"""

import os
import logging
import tempfile
import time
import json
import re
from typing import Dict, List, Any, Optional, Union

# Import specialized libraries for web operations
try:
    import requests
    from bs4 import BeautifulSoup
    import urllib.parse
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException
except ImportError as e:
    logging.error(f"Error importing web processing libraries: {str(e)}")
    logging.warning("Some web processing capabilities may be limited")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebTools:
    """Web Tools class for handling various web operations"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Web Tools with configuration
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.temp_dir = config.get('TEMP_FOLDER', tempfile.gettempdir())
        self.download_dir = config.get('DOWNLOAD_FOLDER', os.path.join(self.temp_dir, 'downloads'))
        self.search_api_key = config.get('SEARCH_API_KEY', '')
        self.search_engine_id = config.get('SEARCH_ENGINE_ID', '')
        self.browser = None
        
        # Ensure directories exist
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.download_dir, exist_ok=True)
        
        logger.info("Web Tools initialized successfully")
    
    def get_available_tools(self) -> List[str]:
        """
        Get a list of all available web tools
        
        Returns:
            List of available tool names
        """
        return [
            "search_web",
            "extract_webpage_content",
            "browse_webpage",
            "download_file",
            "take_screenshot",
            "fill_form"
        ]
    
    def execute_tool(self, instruction: str) -> Dict[str, Any]:
        """
        Execute a web tool based on the instruction
        
        Args:
            instruction: Instruction describing what to do
            
        Returns:
            Dictionary containing the result of the tool execution
        """
        # Parse the instruction to determine which tool to use
        if "بحث في الويب" in instruction or "search web" in instruction.lower():
            query = self._extract_search_query(instruction)
            if query:
                return self.search_web(query)
            else:
                return {"error": "لم يتم تحديد استعلام البحث"}
        
        elif "استخراج محتوى صفحة" in instruction or "extract webpage" in instruction.lower():
            url = self._extract_url(instruction)
            if url:
                return self.extract_webpage_content(url)
            else:
                return {"error": "لم يتم تحديد عنوان URL"}
        
        elif "تصفح صفحة" in instruction or "browse webpage" in instruction.lower():
            url = self._extract_url(instruction)
            if url:
                return self.browse_webpage(url)
            else:
                return {"error": "لم يتم تحديد عنوان URL"}
        
        elif "تنزيل ملف" in instruction or "download file" in instruction.lower():
            url = self._extract_url(instruction)
            output_path = self._extract_output_path(instruction)
            if url:
                return self.download_file(url, output_path)
            else:
                return {"error": "لم يتم تحديد عنوان URL"}
        
        elif "التقاط لقطة شاشة" in instruction or "take screenshot" in instruction.lower():
            url = self._extract_url(instruction)
            output_path = self._extract_output_path(instruction)
            if url:
                return self.take_screenshot(url, output_path)
            else:
                return {"error": "لم يتم تحديد عنوان URL"}
        
        elif "ملء نموذج" in instruction or "fill form" in instruction.lower():
            url = self._extract_url(instruction)
            form_data = self._extract_form_data(instruction)
            if url and form_data:
                return self.fill_form(url, form_data)
            else:
                return {"error": "لم يتم تحديد عنوان URL أو بيانات النموذج"}
        
        else:
            return {"error": "لم يتم التعرف على الأداة المطلوبة"}
    
    def search_web(self, query: str) -> Dict[str, Any]:
        """
        Search the web for the given query
        
        Args:
            query: Search query
            
        Returns:
            Dictionary containing search results
        """
        try:
            # If API keys are available, use Google Custom Search API
            if self.search_api_key and self.search_engine_id:
                return self._search_with_api(query)
            else:
                # Fallback to scraping search results
                return self._search_with_scraping(query)
        except Exception as e:
            logger.error(f"Error searching web: {str(e)}")
            return {"error": f"حدث خطأ أثناء البحث في الويب: {str(e)}"}
    
    def _search_with_api(self, query: str) -> Dict[str, Any]:
        """
        Search using Google Custom Search API
        
        Args:
            query: Search query
            
        Returns:
            Dictionary containing search results
        """
        try:
            # Google Custom Search API endpoint
            url = "https://www.googleapis.com/customsearch/v1"
            
            # Parameters for the API request
            params = {
                "key": self.search_api_key,
                "cx": self.search_engine_id,
                "q": query,
                "num": 10  # Number of results to return
            }
            
            # Make the API request
            response = requests.get(url, params=params)
            data = response.json()
            
            # Extract search results
            results = []
            if "items" in data:
                for item in data["items"]:
                    results.append({
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", "")
                    })
            
            return {
                "query": query,
                "results": results,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error searching with API: {str(e)}")
            return {"error": f"حدث خطأ أثناء البحث باستخدام API: {str(e)}"}
    
    def _search_with_scraping(self, query: str) -> Dict[str, Any]:
        """
        Search by scraping search engine results
        
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
            self.browser.get(f"https://www.google.com/search?q={encoded_query}")
            
            # Wait for results to load
            WebDriverWait(self.browser, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.g"))
            )
            
            # Extract search results
            results = []
            search_results = self.browser.find_elements(By.CSS_SELECTOR, "div.g")
            
            for result in search_results[:10]:  # Limit to first 10 results
                try:
                    title_element = result.find_element(By.CSS_SELECTOR, "h3")
                    title = title_element.text
                    
                    link_element = result.find_element(By.CSS_SELECTOR, "a")
                    link = link_element.get_attribute("href")
                    
                    snippet_element = result.find_element(By.CSS_SELECTOR, "div.VwiC3b")
                    snippet = snippet_element.text
                    
                    results.append({
                        "title": title,
                        "link": link,
                        "snippet": snippet
                    })
                except Exception as e:
                    logger.warning(f"Error extracting search result: {str(e)}")
                    continue
            
            return {
                "query": query,
                "results": results,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error searching with scraping: {str(e)}")
            return {"error": f"حدث خطأ أثناء البحث باستخدام التصفح: {str(e)}"}
    
    def extract_webpage_content(self, url: str) -> Dict[str, Any]:
        """
        Extract content from a webpage
        
        Args:
            url: URL of the webpage
            
        Returns:
            Dictionary containing the extracted content
        """
        try:
            # Make a request to the URL
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title = soup.title.text.strip() if soup.title else ""
            
            # Extract main content
            # This is a simple approach; more sophisticated content extraction may be needed
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
                    href = urllib.parse.urljoin(url, href)
                
                links.append({
                    "href": href,
                    "text": text
                })
            
            return {
                "url": url,
                "title": title,
                "content": main_content,
                "links": links,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error extracting webpage content: {str(e)}")
            return {"error": f"حدث خطأ أثناء استخراج محتوى صفحة الويب: {str(e)}"}
    
    def browse_webpage(self, url: str) -> Dict[str, Any]:
        """
        Browse a webpage using a browser
        
        Args:
            url: URL of the webpage
            
        Returns:
            Dictionary containing information about the browsing session
        """
        try:
            # Initialize browser if not already initialized
            if not self.browser:
                self._initialize_browser()
            
            # Navigate to the URL
            self.browser.get(url)
            
            # Wait for page to load
            time.sleep(3)
            
            # Get page title
            title = self.browser.title
            
            # Get page content
            content = self.browser.page_source
            
            # Parse the content
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract text content
            text_content = soup.get_text(separator='\n', strip=True)
            
            # Extract links
            links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                text = link.get_text(strip=True)
                
                # Convert relative URLs to absolute URLs
                if href.startswith('/'):
                    href = urllib.parse.urljoin(url, href)
                
                links.append({
                    "href": href,
                    "text": text
                })
            
            return {
                "url": url,
                "title": title,
                "text_content": text_content,
                "links": links,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error browsing webpage: {str(e)}")
            return {"error": f"حدث خطأ أثناء تصفح صفحة الويب: {str(e)}"}
    
    def download_file(self, url: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Download a file from the web
        
        Args:
            url: URL of the file to download
            output_path: Path where the file should be saved
            
        Returns:
            Dictionary containing information about the downloaded file
        """
        try:
            # Make a request to the URL
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()
            
            # Get the filename from the URL if not provided
            if not output_path:
                # Try to get filename from Content-Disposition header
                content_disposition = response.headers.get('Content-Disposition')
                if content_disposition:
                    filename_match = re.search(r'filename="?([^"]+)"?', content_disposition)
                    if filename_match:
                        filename = filename_match.group(1)
                    else:
                        filename = os.path.basename(urllib.parse.urlparse(url).path)
                else:
                    filename = os.path.basename(urllib.parse.urlparse(url).path)
                
                # If filename is empty or invalid, use a default name
                if not filename or filename == '':
                    filename = f"download_{int(time.time())}"
                
                output_path = os.path.join(self.download_dir, filename)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Download the file
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return {
                "url": url,
                "file_path": output_path,
                "file_size": os.path.getsize(output_path),
        
(Content truncated due to size limit. Use line ranges to read in chunks)