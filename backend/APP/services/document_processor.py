import pdfplumber
from pdfplumber.utils import extract_text, get_bbox_overlap, obj_to_bbox
import pytesseract
import cv2
import numpy as np
import os
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Process PDF documents to extract text, tables, and images with OCR"""
    
    def __init__(self, upload_dir: str):
        """
        Args:
            upload_dir: Directory path where PDFs are stored
        """
        self.dir = Path(upload_dir)
        self._validate_environment()
        
    def _validate_environment(self):
        """Check required dependencies are available"""
        try:
            pytesseract.get_tesseract_version()
        except EnvironmentError:
            logger.error("Tesseract OCR not installed or not in system PATH")
            raise
            
    def process_all_pdfs(self) -> List[Dict]:
        """Process all PDFs in the directory"""
        if not self.dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.dir}")
            
        all_records = []
        for pdf_file in self.dir.glob("*.pdf"):
            try:
                records = self.process_single_pdf(pdf_file)
                all_records.extend(records)
            except Exception as e:
                logger.error(f"Failed to process {pdf_file}: {str(e)}")
                continue
                
        return all_records

    def process_single_pdf(self, pdf_path: Path) -> List[Dict]:
        """Process a single PDF file"""
        records = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages[:5]):  # Limit to first 5 pages
                try:
                    page_data = self._process_page(page, page_num)
                    records.append({
                        **page_data,
                        "meta": {
                            "source": str(pdf_path),
                            "page": page_num
                        }
                    })
                except Exception as e:
                    logger.error(f"Error processing page {page_num} of {pdf_path}: {str(e)}")
                    continue
        return records

    def _process_page(self, page, page_num: int) -> Dict:
        """Extract all elements from a single page"""
        page_height = page.height
        filtered_page = page
        chars = page.chars
        
        # Process tables
        tables_text = []
        for table in page.find_tables():
            try:
                first_table_char = page.crop(table.bbox).chars[0]
                filtered_page = filtered_page.filter(
                    lambda obj: get_bbox_overlap(obj_to_bbox(obj), table.bbox) is None
                )
                chars = filtered_page.chars
                
                # Convert table to markdown
                df = pd.DataFrame(table.extract())
                if len(df) > 1:  # Has at least header and one row
                    df.columns = df.iloc[0]
                    markdown = df.drop(0).to_markdown(index=False)
                    tables_text.append(markdown)
                    chars.append(first_table_char | {"text": markdown})
            except Exception as e:
                logger.warning(f"Table processing error on page {page_num}: {str(e)}")
                continue
                
        # Extract layout text
        page_text = extract_text(chars, layout=True)
        
        # Process images
        images_text = []
        for img in page.images:
            try:
                image_obj = self._crop_image(page, page_height, img)
                text = self._extract_text_from_image(image_obj.original)
                images_text.append(text)
            except Exception as e:
                logger.warning(f"Image processing error on page {page_num}: {str(e)}")
                continue
                
        return {
            "page_id": page_num,
            "text": page_text,
            "tables": "\n".join(tables_text),
            "images_text": "\n".join(images_text),
            "full_content": f"{page_text}\n\nTables:\n{' '.join(tables_text)}\n\nImages Text:\n{' '.join(images_text)}"
        }

    def _crop_image(self, page, page_height: float, img: Dict):
        """Crop an image from the PDF page"""
        x0 = max(0, img['x0'])
        y0 = max(0, page_height - img['y1'])
        x1 = min(page.width, img['x1'])
        y1 = min(page.height, page_height - img['y0'])
        bbox = (x0, y0, x1, y1)
        return page.crop(bbox).to_image(resolution=400)

    def _extract_text_from_image(self, pil_image) -> str:
        """Perform OCR on an image"""
        try:
            # Convert to OpenCV format
            open_cv_image = np.array(pil_image)[:, :, ::-1]  # RGB to BGR
            
            # Preprocess for OCR
            gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, 
                                    cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
            
            # Improve text detection
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(thresh, kernel, iterations=1)
            
            # Find contours and extract text
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
            
            text_parts = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cropped = open_cv_image[y:y+h, x:x+w]
                text = pytesseract.image_to_string(cropped)
                if text.strip():
                    text_parts.append(text.strip())
                    
            return "\n".join(text_parts)
        except Exception as e:
            logger.error(f"OCR failed: {str(e)}")
            return ""