import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import os
import logging
import json
from typing import List, Dict, Tuple, Optional
import traceback
import tempfile
import hashlib
import math
from collections import defaultdict
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

logger = logging.getLogger(__name__)

class PDFProcessingError(Exception):
    """Custom exception for PDF processing errors"""
    pass

class PDFProcessor:
    def __init__(self, pdf_path: str):
        """Initialize PDF processor with comprehensive error handling"""
        if not os.path.exists(pdf_path):
            raise PDFProcessingError(f"PDF file not found: {pdf_path}")
        
        if not pdf_path.lower().endswith('.pdf'):
            raise PDFProcessingError("File must be a PDF")
        
        # Check file size (max 50MB)
        file_size = os.path.getsize(pdf_path)
        if file_size > 50 * 1024 * 1024:
            raise PDFProcessingError(f"PDF file too large: {file_size / 1024 / 1024:.1f}MB (max 50MB)")
        
        try:
            self.pdf_path = pdf_path
            self.doc = fitz.open(pdf_path)
            
            if self.doc.is_encrypted:
                raise PDFProcessingError("Encrypted PDFs are not supported")
            
            if len(self.doc) == 0:
                raise PDFProcessingError("PDF has no pages")
            
            if len(self.doc) > 200:
                raise PDFProcessingError(f"PDF has too many pages: {len(self.doc)} (max 200)")
                
            self.detected_boxes = []
            self.page_cache = {}
            self.document_structure = None
            self.processing_stats = {
                'total_time': 0,
                'boxes_detected': 0,
                'pages_processed': 0,
                'methods_used': []
            }
            
            # Analyze document structure once
            self.document_structure = self.analyze_document_structure()
            
            logger.info(f"Successfully initialized PDF processor for {pdf_path} with {len(self.doc)} pages")
            
        except fitz.FileDataError as e:
            raise PDFProcessingError(f"Invalid or corrupted PDF file: {str(e)}")
        except Exception as e:
            raise PDFProcessingError(f"Failed to open PDF: {str(e)}")
    
    def analyze_document_structure(self) -> Dict:
        """Analyze overall document layout and structure"""
        structure = {
            'page_dimensions': [],
            'text_density_map': {},
            'common_patterns': [],
            'numbering_patterns': [],
            'layout_consistency': 0.0
        }
        
        try:
            # Sample first few pages to understand structure
            sample_pages = min(3, len(self.doc))
            
            for page_num in range(sample_pages):
                page = self.doc[page_num]
                
                # Store page dimensions
                structure['page_dimensions'].append({
                    'width': page.rect.width,
                    'height': page.rect.height,
                    'page': page_num + 1
                })
                
                # Analyze text patterns
                try:
                    text_dict = page.get_text("dict")
                    self._analyze_text_patterns(text_dict, structure, page_num)
                except Exception as e:
                    logger.warning(f"Could not analyze text patterns for page {page_num + 1}: {str(e)}")
            
            # Calculate layout consistency
            if len(structure['page_dimensions']) > 1:
                widths = [p['width'] for p in structure['page_dimensions']]
                heights = [p['height'] for p in structure['page_dimensions']]
                width_consistency = 1.0 - (np.std(widths) / np.mean(widths)) if np.mean(widths) > 0 else 0
                height_consistency = 1.0 - (np.std(heights) / np.mean(heights)) if np.mean(heights) > 0 else 0
                structure['layout_consistency'] = (width_consistency + height_consistency) / 2
            
            logger.info(f"Document structure analysis complete. Layout consistency: {structure['layout_consistency']:.2f}")
            
        except Exception as e:
            logger.error(f"Error analyzing document structure: {str(e)}")
            
        return structure
    
    def _analyze_text_patterns(self, text_dict: Dict, structure: Dict, page_num: int):
        """Analyze text patterns to identify question numbering and layout"""
        import re
        
        # Common question number patterns
        number_patterns = [
            r'^\d+\.',  # 1. 2. 3.
            r'^\d+\)',  # 1) 2) 3)
            r'^Q\d+',   # Q1 Q2 Q3
            r'^\(\d+\)', # (1) (2) (3)
        ]
        
        # Extract text blocks
        for block in text_dict.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        
                        # Check for question numbering patterns
                        for pattern in number_patterns:
                            if re.match(pattern, text):
                                if pattern not in structure['numbering_patterns']:
                                    structure['numbering_patterns'].append(pattern)
                                break
    
    def detect_all_annotation_types(self) -> List[Dict]:
        """Detect all possible annotation types that could represent boxes"""
        boxes = []
        supported_types = ['Square', 'Rectangle', 'Highlight', 'Ink', 'FreeText', 'Circle', 'Polygon']
        
        try:
            for page_num in range(len(self.doc)):
                try:
                    page = self.doc[page_num]
                    annotations = page.annots()
                    if not annotations:
                        continue
                        
                    for annot in annotations:
                        try:
                            if annot and hasattr(annot, 'type'):
                                annot_type = annot.type[1]
                                
                                if annot_type in supported_types:
                                    box_data = self._process_annotation(annot, page_num, page, annot_type)
                                    if box_data:
                                        boxes.append(box_data)
                                        
                        except Exception as e:
                            logger.warning(f"Error processing annotation on page {page_num + 1}: {str(e)}")
                            continue
                            
                except Exception as e:
                    logger.error(f"Error processing page {page_num + 1} for annotations: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Critical error in annotation detection: {str(e)}")
            
        logger.info(f"Detected {len(boxes)} annotation boxes")
        return boxes
    
    def _process_annotation(self, annot, page_num: int, page, annot_type: str) -> Optional[Dict]:
        """Process individual annotation and extract box data"""
        try:
            rect = annot.rect
            
            # Handle different annotation types
            if annot_type in ['Square', 'Rectangle', 'Highlight']:
                if rect and rect.is_valid and not rect.is_empty:
                    area = (rect.x1 - rect.x0) * (rect.y1 - rect.y0)
                    if area > 100:  # Minimum area threshold
                        return self._create_box_data(rect, page_num, page, 'annotation', 1.0, annot)
                        
            elif annot_type == 'Circle':
                # Convert circle to bounding rectangle
                if rect and rect.is_valid and not rect.is_empty:
                    area = (rect.x1 - rect.x0) * (rect.y1 - rect.y0)
                    if area > 100:
                        return self._create_box_data(rect, page_num, page, 'circle_annotation', 0.9, annot)
                        
            elif annot_type in ['Ink', 'Polygon']:
                # Get vertices and create bounding box
                vertices = annot.vertices
                if vertices and len(vertices) >= 4:
                    x_coords = [v.x for v in vertices]
                    y_coords = [v.y for v in vertices]
                    
                    bbox_rect = fitz.Rect(min(x_coords), min(y_coords), max(x_coords), max(y_coords))
                    area = (bbox_rect.x1 - bbox_rect.x0) * (bbox_rect.y1 - bbox_rect.y0)
                    
                    if area > 100:
                        return self._create_box_data(bbox_rect, page_num, page, f'{annot_type.lower()}_annotation', 0.8, annot)
                        
            elif annot_type == 'FreeText':
                # Text annotations might indicate question areas
                if rect and rect.is_valid and not rect.is_empty:
                    area = (rect.x1 - rect.x0) * (rect.y1 - rect.y0)
                    if area > 500:  # Larger threshold for text annotations
                        return self._create_box_data(rect, page_num, page, 'text_annotation', 0.7, annot)
                        
        except Exception as e:
            logger.debug(f"Error processing {annot_type} annotation: {str(e)}")
            
        return None
    
    def _create_box_data(self, rect, page_num: int, page, box_type: str, confidence: float, annot=None) -> Dict:
        """Create standardized box data structure"""
        box_data = {
            'page': page_num + 1,
            'coordinates': [
                max(0, rect.x0),
                max(0, rect.y0),
                min(page.rect.width, rect.x1),
                min(page.rect.height, rect.y1)
            ],
            'type': box_type,
            'area': (rect.x1 - rect.x0) * (rect.y1 - rect.y0),
            'confidence': confidence
        }
        
        # Add annotation metadata if available
        if annot:
            try:
                metadata = self._extract_annotation_metadata(annot)
                box_data.update(metadata)
            except Exception as e:
                logger.debug(f"Could not extract annotation metadata: {str(e)}")
                
        return box_data
    
    def _extract_annotation_metadata(self, annotation) -> Dict:
        """Extract rich metadata from annotations"""
        metadata = {}
        
        try:
            # Extract color information
            if hasattr(annotation, 'colors') and annotation.colors:
                metadata['color'] = annotation.colors
                
            # Extract opacity
            if hasattr(annotation, 'opacity'):
                metadata['opacity'] = annotation.opacity
                
            # Extract border information
            if hasattr(annotation, 'border'):
                border = annotation.border
                if border:
                    metadata['border_width'] = border.get('width', 1)
                    
            # Extract creation info
            info = annotation.info
            if info:
                metadata['creation_date'] = info.get('creationDate')
                metadata['author'] = info.get('title')
                
        except Exception as e:
            logger.debug(f"Error extracting annotation metadata: {str(e)}")
            
        return metadata
    
    def detect_annotations(self) -> List[Dict]:
        """Legacy method - redirects to enhanced annotation detection"""
        return self.detect_all_annotation_types()
    
    def detect_shapes_opencv(self) -> List[Dict]:
        """Detect rectangular shapes using OpenCV with multiple detection methods"""
        all_boxes = []
        
        try:
            for page_num in range(len(self.doc)):
                try:
                    page = self.doc[page_num]
                    zoom_factor = 2
                    
                    # Convert page to image with error handling
                    try:
                        mat = page.get_pixmap(matrix=fitz.Matrix(zoom_factor, zoom_factor))
                        img_data = mat.tobytes("png")
                    except Exception as e:
                        logger.warning(f"Failed with zoom 2x, trying 1x: {str(e)}")
                        zoom_factor = 1
                        try:
                            mat = page.get_pixmap(matrix=fitz.Matrix(zoom_factor, zoom_factor))
                            img_data = mat.tobytes("png")
                        except Exception as e2:
                            logger.error(f"Failed to render page {page_num + 1}: {str(e2)}")
                            continue
                    
                    # Convert to OpenCV format
                    try:
                        nparr = np.frombuffer(img_data, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if img is None:
                            logger.error(f"Failed to decode image for page {page_num + 1}")
                            continue
                            
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    except Exception as e:
                        logger.error(f"Failed to convert image for page {page_num + 1}: {str(e)}")
                        continue
                    
                    # Enhanced detection methods
                    page_boxes = self._apply_enhanced_detection_methods(gray, page_num, zoom_factor, page, img.shape)
                    all_boxes.extend(page_boxes)
                            
                except Exception as e:
                    logger.error(f"Error processing page {page_num + 1} for shapes: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Critical error in shape detection: {str(e)}")
        
        # Remove duplicates from multiple detection methods
        unique_boxes = self._smart_duplicate_removal(all_boxes)
        logger.info(f"Detected {len(unique_boxes)} unique shape boxes using OpenCV")
        return unique_boxes
    
    def _apply_enhanced_detection_methods(self, gray: np.ndarray, page_num: int, zoom_factor: float, page, img_shape: Tuple) -> List[Dict]:
        """Apply enhanced detection methods with adaptive parameters"""
        all_boxes = []
        
        # Calculate adaptive parameters based on image statistics
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        # Method 1: Enhanced Canny edge detection
        try:
            lower_threshold = max(30, mean_intensity - std_intensity)
            upper_threshold = min(200, mean_intensity + std_intensity)
            
            # Apply Gaussian blur for noise reduction
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(blurred, lower_threshold, upper_threshold, apertureSize=3)
            
            # Morphological operations to connect broken edges
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            contours1, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes1 = self._process_contours_enhanced(contours1, page_num, zoom_factor, page, img_shape, 0.85)
            all_boxes.extend(boxes1)
        except Exception as e:
            logger.warning(f"Enhanced Canny method failed: {str(e)}")
        
        # Method 2: Adaptive threshold with multiple kernels
        try:
            adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            
            # Multiple morphological operations
            kernels = [
                cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            ]
            
            for kernel in kernels:
                try:
                    gradient = cv2.morphologyEx(adaptive, cv2.MORPH_GRADIENT, kernel)
                    closed = cv2.morphologyEx(gradient, cv2.MORPH_CLOSE, kernel)
                    
                    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    boxes = self._process_contours_enhanced(contours, page_num, zoom_factor, page, img_shape, 0.8)
                    all_boxes.extend(boxes)
                except Exception as e:
                    logger.debug(f"Morphological method with kernel failed: {str(e)}")
                    
        except Exception as e:
            logger.warning(f"Adaptive threshold method failed: {str(e)}")
        
        # Method 3: Corner detection for precise rectangles
        try:
            corners = cv2.cornerHarris(gray, 2, 3, 0.04)
            corners = cv2.dilate(corners, None)
            
            threshold = 0.01 * corners.max()
            corner_points = np.where(corners > threshold)
            
            if len(corner_points[0]) > 4:
                rectangles = self._group_corners_into_rectangles(corner_points, img_shape)
                
                for rect in rectangles:
                    x, y, w, h = rect
                    pdf_x = x / zoom_factor
                    pdf_y = y / zoom_factor
                    pdf_w = w / zoom_factor
                    pdf_h = h / zoom_factor
                    
                    if pdf_w > 50 and pdf_h > 30:
                        all_boxes.append({
                            'page': page_num + 1,
                            'coordinates': [pdf_x, pdf_y, pdf_x + pdf_w, pdf_y + pdf_h],
                            'type': 'corner_detection',
                            'area': pdf_w * pdf_h,
                            'confidence': 0.75
                        })
        except Exception as e:
            logger.warning(f"Corner detection method failed: {str(e)}")
        
        # Method 4: Contour-based detection with quality filtering
        try:
            # Binary threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = self._process_contours_enhanced(contours, page_num, zoom_factor, page, img_shape, 0.7)
            all_boxes.extend(boxes)
        except Exception as e:
            logger.warning(f"Contour detection method failed: {str(e)}")
        
        return all_boxes
    
    def _process_contours_enhanced(self, contours: List, page_num: int, zoom_factor: float, page, img_shape: Tuple, base_confidence: float) -> List[Dict]:
        """Enhanced contour processing with better filtering and validation"""
        boxes = []
        
        for contour in contours:
            try:
                area = cv2.contourArea(contour)
                if area < 5000:  # Skip very small contours
                    continue
                
                # Approximate contour
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if it's approximately rectangular
                if 4 <= len(approx) <= 8:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Enhanced size filtering
                    min_width = max(100, img_shape[1] * 0.05)  # At least 5% of page width
                    min_height = max(50, img_shape[0] * 0.02)   # At least 2% of page height
                    max_width_ratio = 0.9
                    max_height_ratio = 0.9
                    
                    if (w > min_width and h > min_height and 
                        w < img_shape[1] * max_width_ratio and 
                        h < img_shape[0] * max_height_ratio):
                        
                        # Calculate quality metrics
                        aspect_ratio = w / h if h > 0 else 0
                        rectangularity = area / (w * h) if w * h > 0 else 0
                        
                        # Enhanced filtering criteria
                        if (0.2 < aspect_ratio < 8.0 and rectangularity > 0.7):
                            # Convert to PDF coordinates
                            pdf_x0 = max(0, x / zoom_factor)
                            pdf_y0 = max(0, y / zoom_factor)
                            pdf_x1 = min(page.rect.width, (x + w) / zoom_factor)
                            pdf_y1 = min(page.rect.height, (y + h) / zoom_factor)
                            
                            # Calculate confidence based on multiple factors
                            confidence = self._calculate_box_confidence(
                                rectangularity, aspect_ratio, area, base_confidence
                            )
                            
                            boxes.append({
                                'page': page_num + 1,
                                'coordinates': [pdf_x0, pdf_y0, pdf_x1, pdf_y1],
                                'type': 'enhanced_contour',
                                'area': (pdf_x1 - pdf_x0) * (pdf_y1 - pdf_y0),
                                'confidence': confidence,
                                'aspect_ratio': aspect_ratio,
                                'rectangularity': rectangularity
                            })
                            
            except Exception as e:
                logger.debug(f"Error processing contour: {str(e)}")
                continue
        
        return boxes
    
    def _calculate_box_confidence(self, rectangularity: float, aspect_ratio: float, area: float, base_confidence: float) -> float:
        """Calculate confidence score based on multiple quality factors"""
        # Rectangularity factor (0.7 to 1.0)
        rect_factor = min(1.0, rectangularity / 0.85)
        
        # Aspect ratio factor (penalize extreme ratios)
        if 0.5 <= aspect_ratio <= 3.0:
            aspect_factor = 1.0
        elif 0.2 <= aspect_ratio <= 6.0:
            aspect_factor = 0.8
        else:
            aspect_factor = 0.6
        
        # Area factor (prefer medium-sized boxes)
        if 10000 <= area <= 100000:
            area_factor = 1.0
        elif 5000 <= area <= 200000:
            area_factor = 0.9
        else:
            area_factor = 0.7
        
        # Combine factors
        confidence = base_confidence * rect_factor * aspect_factor * area_factor
        return min(confidence, 0.95)
    
    def _group_corners_into_rectangles(self, corner_points: Tuple, img_shape: Tuple) -> List[Tuple[int, int, int, int]]:
        """Group corner points into potential rectangles"""
        rectangles = []
        
        try:
            y_coords, x_coords = corner_points
            points = list(zip(x_coords, y_coords))
            
            # Simple clustering approach for grouping corners
            min_distance = 50  # Minimum distance between corners
            
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    for k in range(j + 1, len(points)):
                        for l in range(k + 1, len(points)):
                            # Check if 4 points form a rectangle
                            rect = self._check_rectangle_from_points([points[i], points[j], points[k], points[l]])
                            if rect:
                                x, y, w, h = rect
                                if w > 100 and h > 50:  # Minimum size
                                    rectangles.append(rect)
                                    
        except Exception as e:
            logger.debug(f"Error grouping corners: {str(e)}")
        
        return rectangles
    
    def _check_rectangle_from_points(self, points: List[Tuple[int, int]]) -> Optional[Tuple[int, int, int, int]]:
        """Check if 4 points form a valid rectangle"""
        try:
            # Sort points to find bounding box
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            width = max_x - min_x
            height = max_y - min_y
            
            # Check if points are roughly at corners
            tolerance = min(width, height) * 0.1
            
            expected_corners = [
                (min_x, min_y), (max_x, min_y),
                (min_x, max_y), (max_x, max_y)
            ]
            
            # Check if actual points are close to expected corners
            matches = 0
            for point in points:
                for corner in expected_corners:
                    distance = math.sqrt((point[0] - corner[0])**2 + (point[1] - corner[1])**2)
                    if distance <= tolerance:
                        matches += 1
                        break
            
            if matches >= 3:  # At least 3 corners should match
                return (min_x, min_y, width, height)
                
        except Exception as e:
            logger.debug(f"Error checking rectangle from points: {str(e)}")
        
        return None
    
    def _process_contours(self, contours: List, page_num: int, zoom_factor: float, 
                         page, img_shape: Tuple, base_confidence: float) -> List[Dict]:
        """Process contours to find rectangular boxes"""
        boxes = []
        
        for contour in contours:
            try:
                # Skip very small contours
                area = cv2.contourArea(contour)
                if area < 5000:
                    continue
                    
                # Approximate contour
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if it's approximately rectangular (4-6 vertices)
                if 4 <= len(approx) <= 6:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Filter by size
                    min_width = 100
                    min_height = 50
                    max_width_ratio = 0.9
                    max_height_ratio = 0.9
                    
                    if (w > min_width and h > min_height and 
                        w < img_shape[1] * max_width_ratio and 
                        h < img_shape[0] * max_height_ratio):
                        
                        # Convert to PDF coordinates
                        pdf_x0 = max(0, x / zoom_factor)
                        pdf_y0 = max(0, y / zoom_factor)
                        pdf_x1 = min(page.rect.width, (x + w) / zoom_factor)
                        pdf_y1 = min(page.rect.height, (y + h) / zoom_factor)
                        
                        # Calculate metrics
                        aspect_ratio = w / h if h > 0 else 0
                        rectangularity = area / (w * h) if w * h > 0 else 0
                        
                        # Filter by aspect ratio and rectangularity
                        if 0.3 < aspect_ratio < 5.0 and rectangularity > 0.85:
                            # Adjust confidence based on rectangularity
                            confidence = base_confidence * (0.7 + 0.3 * rectangularity)
                            
                            boxes.append({
                                'page': page_num + 1,
                                'coordinates': [pdf_x0, pdf_y0, pdf_x1, pdf_y1],
                                'type': 'shape',
                                'area': (pdf_x1 - pdf_x0) * (pdf_y1 - pdf_y0),
                                'confidence': min(confidence, 0.95),
                                'aspect_ratio': aspect_ratio,
                                'rectangularity': rectangularity
                            })
                                    
            except Exception as e:
                logger.debug(f"Error processing contour: {str(e)}")
                continue
                
        return boxes
    
    def _detect_rectangles_from_lines(self, lines: np.ndarray, img_shape: Tuple) -> List[Tuple[int, int, int, int]]:
        """Detect rectangles from detected lines using Hough transform"""
        rectangles = []
        
        if lines is None or len(lines) == 0:
            return rectangles
            
        # Group lines into horizontal and vertical with tolerance
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate line properties
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            
            # Skip very short lines
            if length < 30:
                continue
                
            angle = np.abs(np.arctan2(dy, dx) * 180 / np.pi)
            
            # Horizontal lines (angle close to 0 or 180)
            if angle < 15 or angle > 165:
                # Store line with y-coordinate and endpoints
                y_coord = (y1 + y2) / 2
                horizontal_lines.append({
                    'y': y_coord,
                    'x1': min(x1, x2),
                    'x2': max(x1, x2),
                    'line': (x1, y1, x2, y2)
                })
            # Vertical lines (angle close to 90)
            elif 75 < angle < 105:
                # Store line with x-coordinate and endpoints
                x_coord = (x1 + x2) / 2
                vertical_lines.append({
                    'x': x_coord,
                    'y1': min(y1, y2),
                    'y2': max(y1, y2),
                    'line': (x1, y1, x2, y2)
                })
        
        # Find rectangles by matching perpendicular lines
        for h_line in horizontal_lines:
            h_x1, h_y1, h_x2, h_y2 = h_line
            h_y = (h_y1 + h_y2) / 2
            
            for v_line in vertical_lines:
                v_x1, v_y1, v_x2, v_y2 = v_line
                v_x = (v_x1 + v_x2) / 2
                
                # Check if lines could form a rectangle corner
                if (min(h_x1, h_x2) - 20 <= v_x <= max(h_x1, h_x2) + 20 and
                    min(v_y1, v_y2) - 20 <= h_y <= max(v_y1, v_y2) + 20):
                    
                    # Look for matching parallel lines to complete rectangle
                    for h_line2 in horizontal_lines:
                        if h_line == h_line2:
                            continue
                        h2_x1, h2_y1, h2_x2, h2_y2 = h_line2
                        h2_y = (h2_y1 + h2_y2) / 2
                        
                        # Check if second horizontal line is parallel and at reasonable distance
                        if abs(h2_y - h_y) > 50:  # Minimum height
                            for v_line2 in vertical_lines:
                                if v_line == v_line2:
                                    continue
                                v2_x1, v2_y1, v2_x2, v2_y2 = v_line2
                                v2_x = (v2_x1 + v2_x2) / 2
                                
                                # Check if we have a complete rectangle
                                if abs(v2_x - v_x) > 50:  # Minimum width
                                    x = min(v_x, v2_x)
                                    y = min(h_y, h2_y)
                                    w = abs(v2_x - v_x)
                                    h = abs(h2_y - h_y)
                                    
                                    # Validate rectangle dimensions
                                    if 100 < w < img_shape[1] * 0.8 and 100 < h < img_shape[0] * 0.8:
                                        rectangles.append((int(x), int(y), int(w), int(h)))
        
        # Remove duplicate rectangles
        unique_rectangles = []
        for rect in rectangles:
            is_duplicate = False
            for unique_rect in unique_rectangles:
                if (abs(rect[0] - unique_rect[0]) < 20 and 
                    abs(rect[1] - unique_rect[1]) < 20 and
                    abs(rect[2] - unique_rect[2]) < 20 and
                    abs(rect[3] - unique_rect[3]) < 20):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_rectangles.append(rect)
        
        return unique_rectangles
    
    def _filter_overlapping_boxes(self, boxes: List[Dict]) -> List[Dict]:
        """Filter overlapping boxes from multiple detection methods"""
        if not boxes:
            return []
        
        # Sort by confidence (highest first)
        boxes.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        filtered = []
        for box in boxes:
            # Check if this box overlaps significantly with any already selected box
            overlap_found = False
            for selected in filtered:
                if selected['page'] == box['page']:
                    overlap = self.calculate_overlap(box['coordinates'], selected['coordinates'])
                    if overlap > 0.7:  # 70% overlap threshold
                        overlap_found = True
                        break
            
            if not overlap_found:
                filtered.append(box)
        
        return filtered
    
    def detect_all_boxes(self) -> List[Dict]:
        """Detect boxes using all available methods with enhanced processing and error recovery"""
        start_time = time.time()
        
        try:
            # Pre-cache some page renderings for faster processing
            self.cache_page_renderings([1.0, 1.5, 2.0])
            
            # Use error recovery detection system
            all_boxes = self.detect_with_error_recovery()
            
            if not all_boxes:
                logger.error("No boxes detected with any method")
                return []
            
            # Validate and repair any problematic boxes
            repaired_boxes = self.validate_and_repair_boxes(all_boxes)
            
            # Enhanced processing pipeline
            validated_boxes = self._validate_box_content(repaired_boxes)
            sorted_boxes = self.sort_boxes_enhanced(validated_boxes)
            
            # Update processing stats
            self.processing_stats['total_time'] = time.time() - start_time
            self.processing_stats['boxes_detected'] = len(sorted_boxes)
            self.processing_stats['pages_processed'] = len(self.doc)
            self.processing_stats['boxes_repaired'] = len(all_boxes) - len(repaired_boxes)
            self.processing_stats['final_success_rate'] = len(sorted_boxes) / len(all_boxes) if all_boxes else 0
            
            self.detected_boxes = sorted_boxes
            logger.info(f"Final detection: {len(sorted_boxes)} boxes in {self.processing_stats['total_time']:.2f}s")
            logger.info(f"Success rate: {self.processing_stats['final_success_rate']:.1%}")
            
            return sorted_boxes
            
        except Exception as e:
            logger.error(f"Critical error in box detection: {str(e)}")
            self.processing_stats['total_time'] = time.time() - start_time
            self.processing_stats['critical_error'] = str(e)
            return []
    
    def batch_process_pages(self, page_indices: List[int] = None) -> List[Dict]:
        """Process specific pages in batch for memory efficiency"""
        if page_indices is None:
            page_indices = list(range(len(self.doc)))
        
        all_boxes = []
        batch_size = 5  # Process 5 pages at a time for free tier
        
        for i in range(0, len(page_indices), batch_size):
            batch = page_indices[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: pages {batch}")
            
            batch_boxes = []
            
            for page_num in batch:
                try:
                    # Process single page
                    page_boxes = self._process_single_page_all_methods(page_num)
                    batch_boxes.extend(page_boxes)
                    
                except Exception as e:
                    logger.error(f"Error processing page {page_num + 1}: {str(e)}")
                    continue
            
            all_boxes.extend(batch_boxes)
            
            # Memory cleanup after each batch
            gc.collect()
            logger.debug(f"Completed batch {i//batch_size + 1}, total boxes so far: {len(all_boxes)}")
        
        return self._smart_duplicate_removal(all_boxes)
    
    def _process_single_page_all_methods(self, page_num: int) -> List[Dict]:
        """Process a single page with all detection methods"""
        page_boxes = []
        
        try:
            page = self.doc[page_num]
            
            # Method 1: Annotations
            try:
                annotations = page.annots()
                if annotations:
                    for annot in annotations:
                        if annot and hasattr(annot, 'type'):
                            annot_type = annot.type[1]
                            if annot_type in ['Square', 'Rectangle', 'Highlight', 'Ink', 'FreeText', 'Circle', 'Polygon']:
                                box_data = self._process_annotation(annot, page_num, page, annot_type)
                                if box_data:
                                    page_boxes.append(box_data)
            except Exception as e:
                logger.debug(f"Annotation processing failed for page {page_num + 1}: {str(e)}")
            
            # Method 2: Shape detection (if no annotations found)
            if not page_boxes:
                try:
                    shape_boxes = self._process_page_adaptive(page_num)
                    page_boxes.extend(shape_boxes)
                except Exception as e:
                    logger.debug(f"Shape detection failed for page {page_num + 1}: {str(e)}")
            
            # Method 3: Text-based detection (if still no boxes)
            if not page_boxes:
                try:
                    text_dict = page.get_text("dict")
                    text_boxes = self._find_question_text_blocks(text_dict, page_num, page)
                    page_boxes.extend(text_boxes)
                except Exception as e:
                    logger.debug(f"Text detection failed for page {page_num + 1}: {str(e)}")
            
        except Exception as e:
            logger.error(f"Critical error processing page {page_num + 1}: {str(e)}")
        
        return page_boxes
    
    def cache_page_renderings(self, zoom_factors: List[float] = None) -> None:
        """Pre-cache page renderings for faster processing"""
        if zoom_factors is None:
            zoom_factors = [1.0, 1.5, 2.0]
        
        try:
            # Only cache first few pages to avoid memory issues
            max_cache_pages = min(3, len(self.doc))
            
            for page_num in range(max_cache_pages):
                page = self.doc[page_num]
                
                for zoom in zoom_factors:
                    cache_key = f"page_{page_num}_zoom_{zoom}"
                    
                    if cache_key not in self.page_cache:
                        try:
                            mat = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
                            img_data = mat.tobytes("png")
                            
                            # Store in cache (limit cache size)
                            if len(self.page_cache) < 10:  # Max 10 cached images
                                self.page_cache[cache_key] = img_data
                                logger.debug(f"Cached rendering for page {page_num + 1} at zoom {zoom}")
                            
                        except Exception as e:
                            logger.debug(f"Failed to cache page {page_num + 1} at zoom {zoom}: {str(e)}")
                            
        except Exception as e:
            logger.warning(f"Error during page caching: {str(e)}")
    
    def get_cached_page_rendering(self, page_num: int, zoom: float) -> Optional[bytes]:
        """Get cached page rendering if available"""
        cache_key = f"page_{page_num}_zoom_{zoom}"
        return self.page_cache.get(cache_key)
    
    def _smart_duplicate_removal(self, boxes: List[Dict]) -> List[Dict]:
        """Intelligent duplicate removal based on content and geometry"""
        if not boxes:
            return boxes
        
        # Group boxes by page first
        page_groups = defaultdict(list)
        for box in boxes:
            page_groups[box['page']].append(box)
        
        unique_boxes = []
        
        for page_num, page_boxes in page_groups.items():
            # Sort by confidence (highest first)
            page_boxes.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            page_unique = []
            
            for box in page_boxes:
                is_duplicate = False
                
                for existing_box in page_unique:
                    overlap = self.calculate_overlap(box['coordinates'], existing_box['coordinates'])
                    
                    # Enhanced duplicate detection
                    if overlap > 0.7:
                        # If significant overlap, keep the one with higher confidence
                        if box.get('confidence', 0) > existing_box.get('confidence', 0):
                            # Replace existing box with current one
                            page_unique.remove(existing_box)
                            page_unique.append(box)
                        is_duplicate = True
                        break
                    elif overlap > 0.4:
                        # Moderate overlap - check other factors
                        if self._should_merge_boxes(box, existing_box):
                            # Merge boxes
                            merged_box = self._merge_boxes(box, existing_box)
                            page_unique.remove(existing_box)
                            page_unique.append(merged_box)
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    page_unique.append(box)
            
            unique_boxes.extend(page_unique)
        
        return unique_boxes
    
    def _should_merge_boxes(self, box1: Dict, box2: Dict) -> bool:
        """Determine if two boxes should be merged"""
        # Check if boxes are of similar type
        type1 = box1.get('type', '')
        type2 = box2.get('type', '')
        
        # Don't merge annotation boxes with detected boxes unless confidence is low
        if 'annotation' in type1 and 'annotation' not in type2:
            return box2.get('confidence', 0) < 0.6
        
        # Check area similarity
        area1 = box1.get('area', 0)
        area2 = box2.get('area', 0)
        
        if area1 > 0 and area2 > 0:
            area_ratio = min(area1, area2) / max(area1, area2)
            return area_ratio > 0.7
        
        return False
    
    def _merge_boxes(self, box1: Dict, box2: Dict) -> Dict:
        """Merge two overlapping boxes"""
        coords1 = box1['coordinates']
        coords2 = box2['coordinates']
        
        # Create bounding box that encompasses both
        merged_coords = [
            min(coords1[0], coords2[0]),  # min x
            min(coords1[1], coords2[1]),  # min y
            max(coords1[2], coords2[2]),  # max x
            max(coords1[3], coords2[3])   # max y
        ]
        
        # Use higher confidence and better type
        confidence = max(box1.get('confidence', 0), box2.get('confidence', 0))
        
        # Prefer annotation type over detected type
        box_type = box1.get('type', 'merged')
        if 'annotation' in box2.get('type', ''):
            box_type = box2.get('type', 'merged')
        
        merged_box = {
            'page': box1['page'],
            'coordinates': merged_coords,
            'type': f'merged_{box_type}',
            'area': (merged_coords[2] - merged_coords[0]) * (merged_coords[3] - merged_coords[1]),
            'confidence': confidence * 0.9  # Slightly reduce confidence for merged boxes
        }
        
        return merged_box
    
    def _validate_box_content(self, boxes: List[Dict]) -> List[Dict]:
        """Validate boxes based on content quality and characteristics"""
        validated_boxes = []
        
        for box in boxes:
            try:
                # Basic geometric validation
                coords = box['coordinates']
                width = coords[2] - coords[0]
                height = coords[3] - coords[1]
                
                # Skip boxes that are too small or too large
                if width < 30 or height < 20:
                    logger.debug(f"Skipping box - too small: {width}x{height}")
                    continue
                
                # Skip boxes with extreme aspect ratios
                aspect_ratio = width / height if height > 0 else 0
                if aspect_ratio < 0.1 or aspect_ratio > 10:
                    logger.debug(f"Skipping box - extreme aspect ratio: {aspect_ratio}")
                    continue
                
                # Content-based validation (if we can extract the image)
                content_score = self._assess_box_content_quality(box)
                
                # Update confidence based on content assessment
                original_confidence = box.get('confidence', 0.5)
                adjusted_confidence = original_confidence * content_score
                box['confidence'] = adjusted_confidence
                box['content_score'] = content_score
                
                # Only keep boxes with reasonable confidence
                if adjusted_confidence > 0.3:
                    validated_boxes.append(box)
                else:
                    logger.debug(f"Skipping box - low confidence after validation: {adjusted_confidence}")
                    
            except Exception as e:
                logger.debug(f"Error validating box: {str(e)}")
                # Keep box with reduced confidence if validation fails
                box['confidence'] = box.get('confidence', 0.5) * 0.8
                validated_boxes.append(box)
        
        return validated_boxes
    
    def _assess_box_content_quality(self, box: Dict) -> float:
        """Assess the quality of box content without full OCR"""
        try:
            # For now, use geometric properties to assess quality
            coords = box['coordinates']
            width = coords[2] - coords[0]
            height = coords[3] - coords[1]
            area = box.get('area', width * height)
            
            # Size factor - prefer medium-sized boxes
            if 50 <= width <= 400 and 30 <= height <= 300:
                size_factor = 1.0
            elif 30 <= width <= 600 and 20 <= height <= 400:
                size_factor = 0.9
            else:
                size_factor = 0.7
            
            # Area factor
            if 5000 <= area <= 100000:
                area_factor = 1.0
            elif 2000 <= area <= 200000:
                area_factor = 0.9
            else:
                area_factor = 0.8
            
            # Position factor - prefer boxes not at extreme edges
            page_num = box['page'] - 1
            if page_num < len(self.doc):
                page = self.doc[page_num]
                page_width = page.rect.width
                page_height = page.rect.height
                
                # Check if box is too close to edges
                margin_x = coords[0] / page_width
                margin_y = coords[1] / page_height
                
                if 0.05 <= margin_x <= 0.95 and 0.05 <= margin_y <= 0.95:
                    position_factor = 1.0
                else:
                    position_factor = 0.8
            else:
                position_factor = 0.9
            
            # Combine factors
            quality_score = size_factor * area_factor * position_factor
            return min(quality_score, 1.0)
            
        except Exception as e:
            logger.debug(f"Error assessing box content quality: {str(e)}")
            return 0.8  # Default moderate quality
    
    def sort_boxes_enhanced(self, boxes: List[Dict]) -> List[Dict]:
        """Enhanced sorting with better question number assignment"""
        def sort_key(box):
            coords = box['coordinates']
            page = box['page']
            
            # Primary sort by page
            # Secondary sort by vertical position (top to bottom)
            # Tertiary sort by horizontal position (left to right)
            # Add small offset based on confidence to prefer higher confidence boxes
            confidence_offset = box.get('confidence', 0.5) * 0.1
            
            return (page, coords[1] - confidence_offset, coords[0])
        
        sorted_boxes = sorted(boxes, key=sort_key)
        
        # Assign question numbers with gap detection
        for i, box in enumerate(sorted_boxes):
            box['question_number'] = i + 1
            
            # Add sequence validation
            if i > 0:
                prev_box = sorted_boxes[i-1]
                if (box['page'] == prev_box['page'] and 
                    abs(box['coordinates'][1] - prev_box['coordinates'][1]) < 20):
                    # Boxes are on same line - check horizontal order
                    if box['coordinates'][0] < prev_box['coordinates'][0]:
                        logger.warning(f"Potential ordering issue between questions {i} and {i+1}")
        
        return sorted_boxes
    
    def remove_duplicates(self, boxes: List[Dict]) -> List[Dict]:
        """Remove duplicate boxes based on overlap"""
        unique_boxes = []
        
        for box in boxes:
            is_duplicate = False
            for unique_box in unique_boxes:
                if self.calculate_overlap(box['coordinates'], unique_box['coordinates']) > 0.7:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_boxes.append(box)
        
        return unique_boxes
    
    def calculate_overlap(self, box1: List[float], box2: List[float]) -> float:
        """Calculate overlap ratio between two boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)
        
        if x_min >= x_max or y_min >= y_max:
            return 0.0
        
        intersection = (x_max - x_min) * (y_max - y_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        
        return intersection / min(area1, area2)
    
    def sort_boxes(self, boxes: List[Dict]) -> List[Dict]:
        """Sort boxes by page, then by position (top-to-bottom, left-to-right)"""
        def sort_key(box):
            coords = box['coordinates']
            return (box['page'], coords[1], coords[0])  # page, y, x
        
        sorted_boxes = sorted(boxes, key=sort_key)
        
        # Add question numbers
        for i, box in enumerate(sorted_boxes):
            box['question_number'] = i + 1
        
        return sorted_boxes
    
    def extract_question_image(self, box: Dict, output_path: str, zoom: float = 2.0) -> Optional[str]:
        """Extract question image from detected box"""
        page_num = box['page'] - 1
        coords = box['coordinates']
        
        try:
            # Validate page number
            if page_num < 0 or page_num >= len(self.doc):
                logger.error(f"Invalid page number: {page_num + 1}")
                return None
                
            page = self.doc[page_num]
            
            # Validate and adjust coordinates
            coords = [
                max(0, coords[0]),
                max(0, coords[1]),
                min(page.rect.width, coords[2]),
                min(page.rect.height, coords[3])
            ]
            
            # Check if box has valid dimensions
            if coords[2] <= coords[0] or coords[3] <= coords[1]:
                logger.error(f"Invalid box dimensions: {coords}")
                return None
            
            # Create rectangle for cropping
            rect = fitz.Rect(coords[0], coords[1], coords[2], coords[3])
            
            # Adjust zoom if image would be too large
            estimated_size = (coords[2] - coords[0]) * (coords[3] - coords[1]) * zoom * zoom * 3  # RGB bytes
            if estimated_size > 50 * 1024 * 1024:  # 50MB limit
                zoom = min(zoom, np.sqrt(50 * 1024 * 1024 / (estimated_size / (zoom * zoom))))
                logger.warning(f"Adjusted zoom to {zoom:.2f} to avoid memory issues")
            
            # Get pixmap with zoom
            mat = fitz.Matrix(zoom, zoom)
            try:
                pix = page.get_pixmap(matrix=mat, clip=rect)
            except Exception as e:
                logger.warning(f"Failed with zoom {zoom}, trying with zoom 1: {str(e)}")
                mat = fitz.Matrix(1, 1)
                pix = page.get_pixmap(matrix=mat, clip=rect)
            
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save image
            pix.save(output_path)
            
            # Enhance image using PIL
            try:
                img = Image.open(output_path)
                
                # Convert to RGB if needed
                if img.mode not in ['RGB', 'L']:
                    img = img.convert('RGB')
                
                # Apply enhancement only if image is valid
                if img.size[0] > 0 and img.size[1] > 0:
                    from PIL import ImageEnhance, ImageFilter
                    
                    # Denoise
                    img = img.filter(ImageFilter.MedianFilter(size=3))
                    
                    # Enhance contrast
                    enhancer = ImageEnhance.Contrast(img)
                    img = enhancer.enhance(1.3)
                    
                    # Enhance sharpness
                    enhancer = ImageEnhance.Sharpness(img)
                    img = enhancer.enhance(1.2)
                    
                    # Save enhanced image
                    img.save(output_path, quality=95, optimize=True)
                
            except Exception as e:
                logger.warning(f"Could not enhance image, using original: {str(e)}")
            
            logger.info(f"Successfully extracted question image to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to extract question image: {str(e)}")
            return None
    
    def extract_question_image_enhanced(self, box: Dict, output_path: str, zoom: float = 2.0) -> Optional[str]:
        """Enhanced question image extraction with quality optimization"""
        page_num = box['page'] - 1
        coords = box['coordinates']
        
        try:
            # Validate inputs
            if page_num < 0 or page_num >= len(self.doc):
                logger.error(f"Invalid page number: {page_num + 1}")
                return None
                
            page = self.doc[page_num]
            
            # Expand coordinates slightly to capture full question
            padding = 5  # pixels of padding
            expanded_coords = [
                max(0, coords[0] - padding),
                max(0, coords[1] - padding),
                min(page.rect.width, coords[2] + padding),
                min(page.rect.height, coords[3] + padding)
            ]
            
            # Validate dimensions
            if expanded_coords[2] <= expanded_coords[0] or expanded_coords[3] <= expanded_coords[1]:
                logger.error(f"Invalid box dimensions: {expanded_coords}")
                return None
            
            # Adaptive zoom calculation
            box_width = expanded_coords[2] - expanded_coords[0]
            box_height = expanded_coords[3] - expanded_coords[1]
            
            # Calculate optimal zoom for good quality without excessive memory
            target_width = min(1200, max(800, box_width * 2))  # Target width between 800-1200px
            optimal_zoom = target_width / box_width
            zoom = min(zoom, optimal_zoom, 3.0)  # Cap at 3x zoom
            
            # Memory check for free tier
            estimated_size = box_width * box_height * zoom * zoom * 3
            if estimated_size > 20 * 1024 * 1024:  # 20MB limit for free tier
                zoom = min(zoom, math.sqrt(20 * 1024 * 1024 / (box_width * box_height * 3)))
                logger.info(f"Adjusted zoom to {zoom:.2f} for memory efficiency")
            
            # Extract image with multiple fallback options
            extracted_path = self._extract_with_fallback(page, expanded_coords, zoom, output_path)
            if not extracted_path:
                return None
            
            # Apply comprehensive image enhancement
            enhanced_path = self._enhance_question_image(extracted_path, box)
            
            return enhanced_path
            
        except Exception as e:
            logger.error(f"Failed to extract enhanced question image: {str(e)}")
            return None
    
    def _extract_with_fallback(self, page, coords: List[float], zoom: float, output_path: str) -> Optional[str]:
        """Extract image with multiple fallback options"""
        rect = fitz.Rect(coords[0], coords[1], coords[2], coords[3])
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Try different extraction methods
        extraction_methods = [
            (zoom, "high_quality"),
            (max(1.5, zoom * 0.75), "medium_quality"),
            (1.0, "basic_quality")
        ]
        
        for attempt_zoom, quality_level in extraction_methods:
            try:
                mat = fitz.Matrix(attempt_zoom, attempt_zoom)
                pix = page.get_pixmap(matrix=mat, clip=rect)
                
                # Check if extraction was successful
                if pix.width > 0 and pix.height > 0:
                    pix.save(output_path)
                    logger.info(f"Extracted image with {quality_level} at zoom {attempt_zoom:.2f}")
                    return output_path
                    
            except Exception as e:
                logger.warning(f"Extraction failed at zoom {attempt_zoom}: {str(e)}")
                continue
        
        logger.error("All extraction methods failed")
        return None
    
    def _enhance_question_image(self, image_path: str, box: Dict) -> str:
        """Apply comprehensive image enhancement for better OCR"""
        try:
            img = Image.open(image_path)
            
            # Convert to RGB if needed
            if img.mode not in ['RGB', 'L']:
                img = img.convert('RGB')
            
            # Skip enhancement for very small images
            if img.size[0] < 100 or img.size[1] < 50:
                logger.warning("Image too small for enhancement")
                return image_path
            
            # Apply enhancement pipeline
            enhanced_img = self._apply_enhancement_pipeline(img, box)
            
            # Save with optimal settings
            enhanced_img.save(image_path, 
                            format='PNG',  # PNG for better quality
                            optimize=True)
            
            logger.debug(f"Enhanced image saved: {image_path}")
            return image_path
            
        except Exception as e:
            logger.warning(f"Image enhancement failed: {str(e)}")
            return image_path
    
    def _apply_enhancement_pipeline(self, img: Image.Image, box: Dict) -> Image.Image:
        """Apply comprehensive enhancement pipeline"""
        try:
            # Step 1: Noise reduction
            img = img.filter(ImageFilter.MedianFilter(size=3))
            
            # Step 2: Adaptive contrast enhancement
            img_array = np.array(img.convert('L'))
            mean_brightness = np.mean(img_array)
            
            if mean_brightness < 100:  # Dark image
                contrast_factor = 1.4
                brightness_factor = 1.2
            elif mean_brightness > 180:  # Bright image
                contrast_factor = 1.2
                brightness_factor = 0.9
            else:  # Normal image
                contrast_factor = 1.3
                brightness_factor = 1.0
            
            # Apply enhancements
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast_factor)
            
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness_factor)
            
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.2)
            
            return img
            
        except Exception as e:
            logger.warning(f"Enhancement pipeline failed: {str(e)}")
            return img
    
    def get_processing_stats(self) -> Dict:
        """Get processing statistics"""
        return self.processing_stats.copy()
    
    def extract_all_questions_enhanced(self, output_dir: str) -> List[Dict]:
        """Enhanced extraction with instruction filtering and question analysis"""
        extracted_questions = []
        filtered_instructions = []
        
        if not self.detected_boxes:
            logger.warning("No boxes detected. Run detect_all_boxes() first.")
            return extracted_questions
        
        try:
            # Try to import question analyzer for instruction filtering
            try:
                from question_analyzer import QuestionAnalyzer
                analyzer = QuestionAnalyzer()
                use_analyzer = True
                logger.info("QuestionAnalyzer loaded - instruction filtering enabled")
            except ImportError:
                analyzer = None
                use_analyzer = False
                logger.warning("QuestionAnalyzer not available - instruction filtering disabled")
            
            total_boxes = len(self.detected_boxes)
            logger.info(f"Starting enhanced extraction of {total_boxes} detected boxes")
            
            for i, box in enumerate(self.detected_boxes):
                try:
                    question_num = box.get('question_number', i + 1)
                    output_filename = f"question_{question_num}.png"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # Use enhanced extraction method
                    extracted_path = self.extract_question_image_enhanced(box, output_path)
                    
                    if extracted_path:
                        # Analyze for instructions if analyzer is available
                        if use_analyzer:
                            try:
                                analysis = analyzer.analyze_question(extracted_path)
                                
                                # Check if this is an instruction box
                                if analysis.get('is_instruction', False):
                                    logger.info(f"Box {question_num} identified as instruction - filtering out")
                                    filtered_instructions.append({
                                        'box_number': question_num,
                                        'instruction_type': analysis.get('structure', {}).get('instruction_type', 'unknown'),
                                        'confidence': analysis.get('confidence', 0.0),
                                        'text_preview': analysis.get('ocr_text', '')[:100] + '...' if len(analysis.get('ocr_text', '')) > 100 else analysis.get('ocr_text', ''),
                                        'image_path': extracted_path
                                    })
                                    
                                    # Remove the instruction image file to save space
                                    try:
                                        os.remove(extracted_path)
                                        logger.debug(f"Removed instruction image: {extracted_path}")
                                    except Exception as e:
                                        logger.debug(f"Could not remove instruction image: {e}")
                                    
                                    continue  # Skip adding to extracted_questions
                                
                                # Add analysis results for actual questions
                                box.update({
                                    'question_type': analysis.get('question_type', 'Unknown'),
                                    'confidence': analysis.get('confidence', 0.0),
                                    'ocr_text': analysis.get('ocr_text', ''),
                                    'options': analysis.get('options', []),
                                    'structure': analysis.get('structure', {}),
                                    'is_instruction': False
                                })
                                
                            except Exception as e:
                                logger.warning(f"Analysis failed for box {question_num}: {str(e)}")
                                # Continue with extraction even if analysis fails
                                box['is_instruction'] = False
                        else:
                            # No analyzer available - assume it's a question
                            box['is_instruction'] = False
                        
                        box['image_path'] = extracted_path
                        box['extraction_success'] = True
                        extracted_questions.append(box)
                        
                        if use_analyzer and 'question_type' in box:
                            logger.debug(f"Successfully extracted question {question_num}: {box['question_type']} (confidence: {box.get('confidence', 0.0):.2f})")
                        else:
                            logger.debug(f"Successfully extracted question {question_num}")
                    else:
                        logger.warning(f"Failed to extract box {question_num}")
                        box['extraction_success'] = False
                        
                    # Memory cleanup every 10 extractions
                    if (i + 1) % 10 == 0:
                        gc.collect()
                        
                except Exception as e:
                    logger.error(f"Error extracting box {i + 1}: {str(e)}")
                    continue
            
            # Update processing stats
            self.processing_stats['instructions_filtered'] = len(filtered_instructions)
            self.processing_stats['actual_questions'] = len(extracted_questions)
            self.processing_stats['analyzer_used'] = use_analyzer
            
            success_rate = len(extracted_questions) / total_boxes if total_boxes > 0 else 0
            logger.info(f"Enhanced extraction completed: {len(extracted_questions)} questions, {len(filtered_instructions)} instructions filtered out")
            logger.info(f"Success rate: {success_rate:.1%} (excluding filtered instructions)")
            
            if filtered_instructions:
                logger.info("Filtered instruction types:")
                for instr in filtered_instructions:
                    logger.info(f"  - Box {instr['box_number']}: {instr['instruction_type']} (confidence: {instr['confidence']:.2f})")
            
            return extracted_questions
            
        except Exception as e:
            logger.error(f"Critical error during enhanced question extraction: {str(e)}")
            return extracted_questions
    
    def progressive_quality_detection(self) -> List[Dict]:
        """Progressive detection starting with fast methods, refining as needed"""
        all_boxes = []
        
        # Stage 1: Quick annotation scan (fastest)
        logger.info("Stage 1: Quick annotation detection")
        try:
            annotation_boxes = self.detect_annotations()
            if annotation_boxes:
                logger.info(f"Found {len(annotation_boxes)} annotation boxes - skipping other methods")
                return self._smart_duplicate_removal(annotation_boxes)
        except Exception as e:
            logger.warning(f"Annotation detection failed: {str(e)}")
        
        # Stage 2: Low-resolution shape detection
        logger.info("Stage 2: Low-resolution shape detection")
        try:
            low_res_boxes = self._detect_shapes_low_resolution()
            if low_res_boxes:
                all_boxes.extend(low_res_boxes)
                logger.info(f"Found {len(low_res_boxes)} boxes at low resolution")
                
                # If we found reasonable number of boxes, refine them
                if len(low_res_boxes) >= 3:
                    refined_boxes = self._refine_detected_boxes(low_res_boxes)
                    return self._smart_duplicate_removal(refined_boxes)
        except Exception as e:
            logger.warning(f"Low-resolution detection failed: {str(e)}")
        
        # Stage 3: High-resolution detection (if needed)
        if len(all_boxes) < 3:
            logger.info("Stage 3: High-resolution detection")
            try:
                high_res_boxes = self.detect_shapes_opencv()
                all_boxes.extend(high_res_boxes)
            except Exception as e:
                logger.warning(f"High-resolution detection failed: {str(e)}")
        
        # Stage 4: Text-based fallback
        if not all_boxes:
            logger.info("Stage 4: Text-based fallback detection")
            try:
                text_boxes = self.detect_text_based_questions()
                all_boxes.extend(text_boxes)
            except Exception as e:
                logger.warning(f"Text-based detection failed: {str(e)}")
        
        return self._smart_duplicate_removal(all_boxes)
    
    def _detect_shapes_low_resolution(self) -> List[Dict]:
        """Fast, low-resolution shape detection for initial scan"""
        boxes = []
        
        try:
            for page_num in range(len(self.doc)):
                page = self.doc[page_num]
                
                # Use low zoom for speed
                zoom_factor = 1.0
                
                try:
                    mat = page.get_pixmap(matrix=fitz.Matrix(zoom_factor, zoom_factor))
                    img_data = mat.tobytes("png")
                    nparr = np.frombuffer(img_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if img is None:
                        continue
                    
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Simple, fast detection
                    edges = cv2.Canny(gray, 50, 150)
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area > 2000:  # Lower threshold for initial detection
                            x, y, w, h = cv2.boundingRect(contour)
                            
                            # Basic filtering
                            aspect_ratio = w / h if h > 0 else 0
                            if 0.2 < aspect_ratio < 8.0 and w > 50 and h > 30:
                                # Convert to PDF coordinates
                                pdf_x0 = x / zoom_factor
                                pdf_y0 = y / zoom_factor
                                pdf_x1 = (x + w) / zoom_factor
                                pdf_y1 = (y + h) / zoom_factor
                                
                                boxes.append({
                                    'page': page_num + 1,
                                    'coordinates': [pdf_x0, pdf_y0, pdf_x1, pdf_y1],
                                    'type': 'low_res_detection',
                                    'area': w * h / (zoom_factor * zoom_factor),
                                    'confidence': 0.6,
                                    'needs_refinement': True
                                })
                
                except Exception as e:
                    logger.debug(f"Low-res detection failed for page {page_num + 1}: {str(e)}")
                    continue
        
        except Exception as e:
            logger.error(f"Low-resolution detection error: {str(e)}")
        
        return boxes
    
    def _refine_detected_boxes(self, rough_boxes: List[Dict]) -> List[Dict]:
        """Refine roughly detected boxes with higher precision"""
        refined_boxes = []
        
        for box in rough_boxes:
            try:
                if not box.get('needs_refinement', False):
                    refined_boxes.append(box)
                    continue
                
                page_num = box['page'] - 1
                coords = box['coordinates']
                
                # Expand the area slightly for refinement
                padding = 20
                expanded_coords = [
                    max(0, coords[0] - padding),
                    max(0, coords[1] - padding),
                    coords[2] + padding,
                    coords[3] + padding
                ]
                
                # Extract region at higher resolution
                page = self.doc[page_num]
                rect = fitz.Rect(expanded_coords[0], expanded_coords[1], 
                               expanded_coords[2], expanded_coords[3])
                
                # Higher zoom for refinement
                zoom_factor = 2.0
                mat = page.get_pixmap(matrix=fitz.Matrix(zoom_factor, zoom_factor), clip=rect)
                img_data = mat.tobytes("png")
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # More precise detection on the cropped region
                    edges = cv2.Canny(gray, 30, 90)
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    best_contour = None
                    best_area = 0
                    
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area > best_area and area > 1000:
                            best_contour = contour
                            best_area = area
                    
                    if best_contour is not None:
                        x, y, w, h = cv2.boundingRect(best_contour)
                        
                        # Convert back to PDF coordinates
                        refined_coords = [
                            expanded_coords[0] + x / zoom_factor,
                            expanded_coords[1] + y / zoom_factor,
                            expanded_coords[0] + (x + w) / zoom_factor,
                            expanded_coords[1] + (y + h) / zoom_factor
                        ]
                        
                        refined_box = box.copy()
                        refined_box['coordinates'] = refined_coords
                        refined_box['type'] = 'refined_detection'
                        refined_box['confidence'] = 0.8
                        refined_box['needs_refinement'] = False
                        refined_box['area'] = w * h / (zoom_factor * zoom_factor)
                        
                        refined_boxes.append(refined_box)
                    else:
                        # Keep original if refinement failed
                        box['confidence'] = 0.5
                        refined_boxes.append(box)
                else:
                    # Keep original if image processing failed
                    refined_boxes.append(box)
                    
            except Exception as e:
                logger.debug(f"Box refinement failed: {str(e)}")
                # Keep original box if refinement fails
                refined_boxes.append(box)
        
        return refined_boxes
    
    def detect_with_error_recovery(self) -> List[Dict]:
        """Detect boxes with comprehensive error recovery"""
        recovery_strategies = [
            ("progressive_quality", self.progressive_quality_detection),
            ("confidence_threshold", lambda: self.detect_boxes_with_confidence_threshold(0.3)),
            ("batch_processing", lambda: self.batch_process_pages()),
            ("text_based_only", self.detect_text_based_questions)
        ]
        
        for strategy_name, strategy_func in recovery_strategies:
            try:
                logger.info(f"Trying detection strategy: {strategy_name}")
                boxes = strategy_func()
                
                if boxes:
                    logger.info(f"Success with {strategy_name}: found {len(boxes)} boxes")
                    self.processing_stats['successful_strategy'] = strategy_name
                    return boxes
                else:
                    logger.warning(f"Strategy {strategy_name} found no boxes")
                    
            except Exception as e:
                logger.error(f"Strategy {strategy_name} failed: {str(e)}")
                continue
        
        logger.error("All detection strategies failed")
        return []
    
    def validate_and_repair_boxes(self, boxes: List[Dict]) -> List[Dict]:
        """Validate and attempt to repair problematic boxes"""
        repaired_boxes = []
        
        for box in boxes:
            try:
                coords = box['coordinates']
                
                # Basic validation
                if len(coords) != 4:
                    logger.debug("Skipping box with invalid coordinates length")
                    continue
                
                width = coords[2] - coords[0]
                height = coords[3] - coords[1]
                
                # Repair negative dimensions
                if width <= 0 or height <= 0:
                    logger.debug("Attempting to repair box with negative dimensions")
                    # Try to fix by swapping coordinates
                    fixed_coords = [
                        min(coords[0], coords[2]),
                        min(coords[1], coords[3]),
                        max(coords[0], coords[2]),
                        max(coords[1], coords[3])
                    ]
                    
                    fixed_width = fixed_coords[2] - fixed_coords[0]
                    fixed_height = fixed_coords[3] - fixed_coords[1]
                    
                    if fixed_width > 0 and fixed_height > 0:
                        box['coordinates'] = fixed_coords
                        box['area'] = fixed_width * fixed_height
                        box['confidence'] = box.get('confidence', 0.5) * 0.8  # Reduce confidence
                        logger.debug("Successfully repaired box dimensions")
                    else:
                        logger.debug("Could not repair box, skipping")
                        continue
                
                # Validate page bounds
                page_num = box.get('page', 1) - 1
                if 0 <= page_num < len(self.doc):
                    page = self.doc[page_num]
                    page_width = page.rect.width
                    page_height = page.rect.height
                    
                    # Clamp coordinates to page bounds
                    clamped_coords = [
                        max(0, min(coords[0], page_width)),
                        max(0, min(coords[1], page_height)),
                        max(0, min(coords[2], page_width)),
                        max(0, min(coords[3], page_height))
                    ]
                    
                    if clamped_coords != coords:
                        box['coordinates'] = clamped_coords
                        box['confidence'] = box.get('confidence', 0.5) * 0.9
                        logger.debug("Clamped box coordinates to page bounds")
                
                repaired_boxes.append(box)
                
            except Exception as e:
                logger.debug(f"Error validating/repairing box: {str(e)}")
                continue
        
        logger.info(f"Validated and repaired {len(repaired_boxes)}/{len(boxes)} boxes")
        return repaired_boxes
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'doc') and self.doc:
                self.doc.close()
            self.page_cache.clear()
            gc.collect()
            logger.info("PDF processor cleanup completed")
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")
        
    def detect_text_based_questions(self) -> List[Dict]:
        """Detect questions based on text patterns when no annotations/shapes found"""
        boxes = []
        
        try:
            for page_num in range(len(self.doc)):
                page = self.doc[page_num]
                
                # Get text with position information
                text_dict = page.get_text("dict")
                question_blocks = self._find_question_text_blocks(text_dict, page_num, page)
                boxes.extend(question_blocks)
                
        except Exception as e:
            logger.error(f"Error in text-based question detection: {str(e)}")
        
        logger.info(f"Detected {len(boxes)} text-based question boxes")
        return boxes
    
    def _find_question_text_blocks(self, text_dict: Dict, page_num: int, page) -> List[Dict]:
        """Find question blocks based on text patterns and layout"""
        question_blocks = []
        
        try:
            import re
            
            # Question number patterns
            question_patterns = [
                r'^\s*(\d+)\.\s*',  # 1. 2. 3.
                r'^\s*(\d+)\)\s*',  # 1) 2) 3)
                r'^\s*Q\.?\s*(\d+)',  # Q1 Q.1
                r'^\s*\((\d+)\)\s*',  # (1) (2) (3)
            ]
            
            blocks = text_dict.get("blocks", [])
            current_question = None
            
            for block in blocks:
                if "lines" not in block:
                    continue
                
                block_text = ""
                block_bbox = block.get("bbox", [0, 0, 0, 0])
                
                # Extract text from block
                for line in block["lines"]:
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text:
                            block_text += text + " "
                
                block_text = block_text.strip()
                
                # Check if this block starts a new question
                for pattern in question_patterns:
                    match = re.match(pattern, block_text)
                    if match:
                        # If we have a previous question, finalize it
                        if current_question:
                            question_blocks.append(current_question)
                        
                        # Start new question
                        current_question = {
                            'page': page_num + 1,
                            'coordinates': list(block_bbox),
                            'type': 'text_based',
                            'area': (block_bbox[2] - block_bbox[0]) * (block_bbox[3] - block_bbox[1]),
                            'confidence': 0.6,
                            'question_number': int(match.group(1)),
                            'text_content': block_text[:100]  # First 100 chars for reference
                        }
                        break
                else:
                    # If this block doesn't start a question but we have a current question,
                    # extend the current question's bounding box
                    if current_question and block_text:
                        # Check if this block is close to the current question
                        curr_bbox = current_question['coordinates']
                        vertical_distance = block_bbox[1] - curr_bbox[3]
                        
                        # If blocks are close vertically (within 50 points), extend the question
                        if 0 <= vertical_distance <= 50:
                            # Extend bounding box
                            current_question['coordinates'] = [
                                min(curr_bbox[0], block_bbox[0]),
                                min(curr_bbox[1], block_bbox[1]),
                                max(curr_bbox[2], block_bbox[2]),
                                max(curr_bbox[3], block_bbox[3])
                            ]
                            # Update area
                            coords = current_question['coordinates']
                            current_question['area'] = (coords[2] - coords[0]) * (coords[3] - coords[1])
            
            # Don't forget the last question
            if current_question:
                question_blocks.append(current_question)
                
        except Exception as e:
            logger.error(f"Error finding question text blocks: {str(e)}")
        
        return question_blocks
    
    def detect_boxes_with_confidence_threshold(self, min_confidence: float = 0.5) -> List[Dict]:
        """Detect boxes with a minimum confidence threshold"""
        all_methods_boxes = []
        
        # Try all detection methods
        try:
            annotation_boxes = self.detect_annotations()
            all_methods_boxes.extend(annotation_boxes)
        except Exception as e:
            logger.warning(f"Annotation detection failed: {str(e)}")
        
        try:
            shape_boxes = self.detect_shapes_opencv()
            all_methods_boxes.extend(shape_boxes)
        except Exception as e:
            logger.warning(f"Shape detection failed: {str(e)}")
        
        # If no boxes found with traditional methods, try text-based detection
        if not all_methods_boxes:
            try:
                text_boxes = self.detect_text_based_questions()
                all_methods_boxes.extend(text_boxes)
                logger.info("Fallback to text-based detection successful")
            except Exception as e:
                logger.warning(f"Text-based detection failed: {str(e)}")
        
        # Filter by confidence
        filtered_boxes = [box for box in all_methods_boxes 
                         if box.get('confidence', 0) >= min_confidence]
        
        return self._smart_duplicate_removal(filtered_boxes)
    
    def auto_detect_optimal_parameters(self) -> Dict:
        """Automatically detect optimal parameters for this specific PDF"""
        params = {
            'zoom_factor': 2.0,
            'confidence_threshold': 0.5,
            'min_box_area': 1000,
            'max_box_area': 200000,
            'preferred_methods': []
        }
        
        try:
            # Analyze document characteristics
            total_pages = len(self.doc)
            avg_page_area = np.mean([p.rect.width * p.rect.height for p in self.doc[:min(3, total_pages)]])
            
            # Adjust parameters based on document size
            if avg_page_area > 500000:  # Large pages
                params['zoom_factor'] = 1.5
                params['min_box_area'] = 2000
            elif avg_page_area < 200000:  # Small pages
                params['zoom_factor'] = 2.5
                params['min_box_area'] = 500
            
            # Test different methods on first page to see which works best
            if total_pages > 0:
                test_results = self._test_detection_methods_on_sample()
                params['preferred_methods'] = test_results['best_methods']
                params['confidence_threshold'] = test_results['optimal_threshold']
            
            logger.info(f"Auto-detected optimal parameters: {params}")
            
        except Exception as e:
            logger.warning(f"Could not auto-detect parameters, using defaults: {str(e)}")
        
        return params
    
    def _test_detection_methods_on_sample(self) -> Dict:
        """Test different detection methods on a sample page"""
        results = {
            'best_methods': [],
            'optimal_threshold': 0.5,
            'method_scores': {}
        }
        
        try:
            # Test on first page only
            sample_page = 0
            
            # Test annotation detection
            try:
                page = self.doc[sample_page]
                annotations = page.annots()
                if annotations and len(list(annotations)) > 0:
                    results['method_scores']['annotations'] = 0.9
                    results['best_methods'].append('annotations')
            except Exception:
                results['method_scores']['annotations'] = 0.0
            
            # Test shape detection (simplified)
            try:
                page = self.doc[sample_page]
                mat = page.get_pixmap(matrix=fitz.Matrix(1, 1))
                img_data = mat.tobytes("png")
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 50, 150)
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Score based on number of reasonable contours found
                    reasonable_contours = [c for c in contours if cv2.contourArea(c) > 1000]
                    if len(reasonable_contours) > 0:
                        results['method_scores']['opencv'] = min(0.8, len(reasonable_contours) / 10)
                        results['best_methods'].append('opencv')
            except Exception:
                results['method_scores']['opencv'] = 0.0
            
            # Test text-based detection
            try:
                text_dict = self.doc[sample_page].get_text("dict")
                question_blocks = self._find_question_text_blocks(text_dict, sample_page, self.doc[sample_page])
                if len(question_blocks) > 0:
                    results['method_scores']['text_based'] = 0.6
                    results['best_methods'].append('text_based')
            except Exception:
                results['method_scores']['text_based'] = 0.0
            
            # Determine optimal threshold based on method reliability
            if results['method_scores'].get('annotations', 0) > 0.8:
                results['optimal_threshold'] = 0.7  # High threshold for reliable annotations
            elif results['method_scores'].get('opencv', 0) > 0.5:
                results['optimal_threshold'] = 0.5  # Medium threshold for shape detection
            else:
                results['optimal_threshold'] = 0.3  # Low threshold for text-based fallback
                
        except Exception as e:
            logger.warning(f"Error testing detection methods: {str(e)}")
        
        return results
    
    def extract_all_questions(self, output_dir: str) -> List[Dict]:
        """Extract all detected questions as images"""
        extracted_questions = []
        
        for box in self.detected_boxes:
            question_num = box['question_number']
            filename = f"question_{question_num}.png"
            output_path = os.path.join(output_dir, filename)
            
            try:
                self.extract_question_image(box, output_path)
                
                extracted_questions.append({
                    'question_number': question_num,
                    'image_path': output_path,
                    'box_coordinates': json.dumps(box['coordinates']),
                    'page_number': box['page']
                })
                
            except Exception as e:
                logger.error(f"Error extracting question {question_num}: {e}")
        
        return extracted_questions
    
    def get_pdf_info(self) -> Dict:
        """Get PDF information"""
        return {
            'total_pages': len(self.doc),
            'metadata': self.doc.metadata,
            'total_boxes_detected': len(self.detected_boxes)
        }
    
    def close(self):
        """Close PDF document"""
        if self.doc:
            self.doc.close()
