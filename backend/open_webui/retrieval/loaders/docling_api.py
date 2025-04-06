import requests
import logging
import os
import sys
from typing import List, Dict, Any

from langchain_core.documents import Document
from open_webui.env import SRC_LOG_LEVELS, GLOBAL_LOG_LEVEL

logging.basicConfig(stream=sys.stdout, level=GLOBAL_LOG_LEVEL)
log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])


class DoclingApiLoader:
    """
    Loads documents by processing them through the Docling API's synchronous conversion endpoint.
    """

    def __init__(self, url: str, file_path: str, extract_tables_as_images: bool = True, image_resolution_scale: int = 4):
        """
        Initializes the loader.

        Args:
            url: The base URL of the Docling API server (e.g., http://localhost:8822)
            file_path: The local path to the document file to process.
            extract_tables_as_images: Whether to extract tables as images (default: True)
            image_resolution_scale: Control the resolution of extracted images (1-4, default: 4)
        """
        if not url:
            raise ValueError("Docling API URL cannot be empty.")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at {file_path}")
        
        # Ensure URL doesn't end with a slash
        self.url = url.rstrip("/")
        self.file_path = file_path
        self.extract_tables_as_images = extract_tables_as_images
        self.image_resolution_scale = max(1, min(4, image_resolution_scale))  # Ensure value is between 1-4

    def load(self) -> List[Document]:
        """
        Executes the document conversion through the Docling API.

        Returns:
            A list of Document objects containing the processed content.
        """
        try:
            # Prepare the API request
            endpoint = f"{self.url}/documents/convert"
            
            # Determine file mime type based on extension (basic implementation)
            file_name = os.path.basename(self.file_path)
            file_ext = file_name.split(".")[-1].lower()
            mime_type = self._get_mime_type(file_ext)
            
            # Prepare the multipart form data
            with open(self.file_path, "rb") as f:
                files = {
                    "document": (file_name, f, mime_type)
                }
                
                data = {
                    "extract_tables_as_images": str(self.extract_tables_as_images).lower(),
                    "image_resolution_scale": str(self.image_resolution_scale)
                }
                
                log.info(f"Sending document to Docling API for conversion: {file_name}")
                response = requests.post(endpoint, files=files, data=data)
            
            # Process the response
            if response.ok:
                result = response.json()
                log.info("Document conversion successful")
                log.debug(f"Docling API response: {result}")
                
                # Extract content from the response
                # Try different possible response structures
                content = ""
                
                # Check for different possible response structures
                if "content" in result:
                    content = result["content"]
                elif "text" in result:
                    content = result["text"]
                elif "markdown" in result:
                    content = result["markdown"]
                elif "md_content" in result:
                    content = result["md_content"]
                
                # If we still don't have content, check if there's a document object
                if not content and "document" in result:
                    document = result["document"]
                    if isinstance(document, dict):
                        content = document.get("content", document.get("text", document.get("markdown", document.get("md_content", ""))))
                
                if not content:
                    log.warning("No content found in Docling API response")
                    content = "<No text content found>"
                    
                log.debug(f"Extracted content: {content[:100]}...")  # Log first 100 chars
                
                # Create metadata from the response
                metadata = {
                    "source": file_name,
                    "content_type": mime_type,
                }
                
                # Add any additional metadata from the response
                if "metadata" in result:
                    metadata.update(result["metadata"])
                
                return [Document(page_content=content, metadata=metadata)]
            else:
                error_msg = f"Error calling Docling API: {response.status_code} - {response.reason}"
                if response.text:
                    try:
                        error_data = response.json()
                        if "detail" in error_data:
                            error_msg += f" - {error_data['detail']}"
                    except Exception:
                        error_msg += f" - {response.text}"
                
                log.error(error_msg)
                return [Document(page_content=f"Error during processing: {error_msg}", metadata={})]
                
        except Exception as e:
            log.error(f"An error occurred during the loading process: {e}")
            return [Document(page_content=f"Error during processing: {e}", metadata={})]
    
    def _get_mime_type(self, file_ext: str) -> str:
        """
        Determines the MIME type based on file extension.
        
        Args:
            file_ext: The file extension
            
        Returns:
            The corresponding MIME type or a default value
        """
        mime_types = {
            "pdf": "application/pdf",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "doc": "application/msword",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "xls": "application/vnd.ms-excel",
            "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "ppt": "application/vnd.ms-powerpoint",
            "txt": "text/plain",
            "csv": "text/csv",
            "html": "text/html",
            "htm": "text/html",
            "xml": "application/xml",
            "json": "application/json",
            "md": "text/markdown",
        }
        
        return mime_types.get(file_ext, "application/octet-stream")
