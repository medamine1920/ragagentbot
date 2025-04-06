import fitz  # PyMuPDF
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image
import io
import cv2
import numpy as np
import logging
from typing import List, Union
from langchain.schema import Document

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, enhance_ocr: bool = True):
        self.enhance_ocr = enhance_ocr

    def process_file(self, file_bytes: bytes, filename: str) -> List[Document]:
        """Process either PDF or image file"""
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return self._process_image(file_bytes, filename)
        elif filename.lower().endswith('.pdf'):
            return self._process_pdf(file_bytes, filename)
        else:
            logger.warning(f"Unsupported file type: {filename}")
            return []

    def _process_pdf(self, pdf_bytes: bytes, filename: str) -> List[Document]:
        """Process PDF file with text and image extraction"""
        docs = []
        
        try:
            # Extract text using PyMuPDF
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                for page_num, page in enumerate(doc):
                    text = page.get_text()
                    metadata = {
                        "source": filename,
                        "page": page_num + 1,
                        "type": "pdf"
                    }

                    # If no text found, try OCR on page images
                    if not text.strip():
                        images = convert_from_bytes(pdf_bytes, first_page=page_num+1, last_page=page_num+1)
                        if images:
                            text = self._enhance_and_ocr(images[0])
                            metadata["ocr"] = True

                    docs.append(Document(
                        page_content=text,
                        metadata=metadata
                    ))
        except Exception as e:
            logger.error(f"Error processing PDF {filename}: {str(e)}")
        
        return docs

    def _process_image(self, image_bytes: bytes, filename: str) -> List[Document]:
        """Process image file with OCR"""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            text = self._enhance_and_ocr(image)
            
            return [Document(
                page_content=text,
                metadata={
                    "source": filename,
                    "type": "image"
                }
            )]
        except Exception as e:
            logger.error(f"Error processing image {filename}: {str(e)}")
            return []

    def _enhance_and_ocr(self, image: Image.Image) -> str:
        """Enhance image and perform OCR"""
        if self.enhance_ocr:
            # Convert to OpenCV format
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Preprocessing
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            _, threshold = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Convert back to PIL Image
            enhanced_image = Image.fromarray(threshold)
        else:
            enhanced_image = image

        return pytesseract.image_to_string(enhanced_image, lang='eng')