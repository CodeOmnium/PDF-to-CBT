import cv2
import numpy as np
from PIL import Image
import pytesseract
import re
import logging
from typing import Dict, List, Tuple, Optional
import spacy
from collections import Counter
import traceback
import os

logger = logging.getLogger(__name__)

class QuestionAnalysisError(Exception):
    """Custom exception for question analysis errors"""
    pass

class QuestionAnalyzer:
    def __init__(self):
        """Initialize QuestionAnalyzer with robust error handling"""
        # Load spaCy model for text processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except OSError as e:
            logger.warning(f"spaCy model not found: {str(e)}. Using basic text processing.")
            self.nlp = None
        except Exception as e:
            logger.error(f"Unexpected error loading spaCy: {str(e)}")
            self.nlp = None
        
        # Configure Tesseract with error handling
        try:
            # Check if Tesseract is installed
            import subprocess
            result = subprocess.run(['tesseract', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                logger.info(f"Tesseract version: {result.stdout.split()[1] if result.stdout else 'Unknown'}")
            else:
                logger.warning("Tesseract not properly installed")
        except Exception as e:
            logger.warning(f"Could not verify Tesseract installation: {str(e)}")
        
        # Instruction patterns to filter out (these should NOT be treated as questions)
        self.instruction_patterns = self._get_instruction_patterns()
        
        # Enhanced question type patterns with weights
        self.patterns = {
            'scq': {
                'patterns': [
                    (r'\([A-Da-d]\)', 0.8),  # Options (A), (B), (C), (D)
                    (r'[A-Da-d][\.\)]', 0.8),  # A. B. C. D. or A) B) C) D)
                    (r'only\s+one\s+correct', 0.9),
                    (r'single\s+correct', 0.9),
                    (r'choose\s+the\s+correct', 0.7),
                    (r'which\s+one\s+of\s+the\s+following', 0.6),
                    (r'correct\s+option\s+is', 0.8),
                    (r'incorrect\s+statement', 0.7),
                    (r'false\s+statement', 0.7)
                ],
                'negative_patterns': [
                    r'one\s+or\s+more',
                    r'multiple\s+correct',
                    r'column',
                    r'match',
                    r'integer'
                ]
            },
            'mcq': {
                'patterns': [
                    (r'one\s+or\s+more\s+correct', 0.95),
                    (r'multiple\s+correct', 0.95),
                    (r'more\s+than\s+one\s+correct', 0.95),
                    (r'which\s+of\s+the\s+following.*correct', 0.8),
                    (r'select\s+all\s+that\s+apply', 0.9),
                    (r'all\s+correct\s+answers', 0.9),
                    (r'multiple\s+options', 0.8),
                    (r'choose\s+all\s+correct', 0.9)
                ],
                'negative_patterns': [
                    r'only\s+one',
                    r'single\s+correct',
                    r'column',
                    r'match'
                ]
            },
            'integer': {
                'patterns': [
                    (r'nearest\s+integer', 0.95),
                    (r'integer\s+value', 0.95),
                    (r'answer.*\d+', 0.7),
                    (r'range.*\d+.*to.*\d+', 0.8),
                    (r'0.*to.*9999', 0.9),
                    (r'numerical\s+value', 0.8),
                    (r'digit\s+integer', 0.9),
                    (r'whole\s+number', 0.8),
                    (r'numeric\s+answer', 0.8),
                    (r'answer\s+is\s+\d+', 0.85),
                    (r'sum.*is', 0.6),
                    (r'product.*is', 0.6),
                    (r'difference.*is', 0.6)
                ],
                'negative_patterns': [
                    r'\([A-Da-d]\)',
                    r'[A-Da-d][\.\)]',
                    r'column',
                    r'match'
                ]
            },
            'match_column': {
                'patterns': [
                    (r'list[\s\-]*i.*list[\s\-]*ii', 0.98),  # List-I and List-II
                    (r'column[\s\-]*i.*column[\s\-]*ii', 0.95),  # Column-I and Column-II
                    (r'match.*list', 0.95),  # Match the list
                    (r'match.*column', 0.95),  # Match the column
                    (r'matching.*list', 0.95),  # Matching list sets
                    (r'\([P-Sp-s]\).*\([1-51-5]\)', 0.9),  # (P), (Q), (R), (S) with (1), (2), (3), (4), (5)
                    (r'[P-Sp-s][\s\-]*[1-51-5]', 0.85),  # P-1, Q-2 type patterns
                    (r'codes?\s*:?\s*\([A-Da-d]\)', 0.9),  # Codes: (A), (B), (C), (D)
                    (r'combination.*correct', 0.85),  # Correct combination
                    (r'four\s+entries.*five\s+entries', 0.95),  # Four entries... five entries
                    (r'only\s+one.*four\s+options', 0.9),  # Only one of these four options
                    (r'satisfies.*condition', 0.8),  # Satisfies the condition
                    (r'multiple\s+choice\s+question.*list', 0.9)  # Multiple choice question based on lists
                ],
                'negative_patterns': [
                    r'one\s+or\s+more\s+correct',
                    r'multiple\s+correct',
                    r'integer\s+value',
                    r'numerical\s+value'
                ]
            }
        }
        
        # Instruction detection patterns to filter out instruction boxes
        self.instruction_patterns = self._get_instruction_patterns()
    
    def _get_instruction_patterns(self) -> Dict[str, List[Tuple[str, float]]]:
        """Get patterns to identify instruction boxes that should be filtered out"""
        return {
            'section_instructions': [
                (r'section[\s\-]*\d+.*this\s+section\s+contains', 0.95),
                (r'section[\s\-]*[a-z].*this\s+section\s+contains', 0.95),
                (r'part[\s\-]*\d+.*this\s+part\s+contains', 0.95),
                (r'this\s+section\s+contains.*questions?', 0.9),
                (r'maximum\s+marks?\s*:?\s*\d+', 0.85),
                (r'time\s*:?\s*\d+\s+hours?', 0.85),
                (r'each\s+question\s+has\s+four\s+options', 0.9),
                (r'only\s+one\s+of\s+these\s+four\s+options', 0.9)
            ],
            'marking_scheme_instructions': [
                (r'marking\s+scheme\s*:?', 0.95),
                (r'full\s+marks?\s*:?\s*\+?\d+', 0.9),
                (r'zero\s+marks?\s*:?\s*0', 0.9),
                (r'negative\s+marks?\s*:?\s*[\-\u2212]?\d+', 0.9),
                (r'if\s+none\s+of\s+the\s+options\s+is\s+chosen', 0.95),
                (r'question\s+is\s+unanswered', 0.9),
                (r'in\s+all\s+other\s+cases', 0.85),
                (r'answer.*will\s+be\s+evaluated\s+according', 0.9)
            ],
            'question_format_instructions': [
                (r'each\s+set\s+has\s+one\s+multiple\s+choice', 0.95),
                (r'each\s+set\s+has\s+two\s+lists', 0.95),
                (r'list[\s\-]*i\s+has\s+four\s+entries', 0.95),
                (r'list[\s\-]*ii\s+has\s+five\s+entries', 0.95),
                (r'\([p-s]\).*\([1-5]\)', 0.9),
                (r'four\s+options\s+are\s+given', 0.9),
                (r'satisfies\s+the\s+condition\s+asked', 0.9),
                (r'multiple\s+choice\s+question\s+based\s+on', 0.9)
            ],
            'general_instructions': [
                (r'instructions?\s*:?', 0.8),
                (r'read\s+the\s+following\s+instructions', 0.9),
                (r'before\s+attempting\s+the\s+questions?', 0.8),
                (r'choose\s+the\s+option\s+corresponding', 0.8),
                (r'for\s+each\s+question.*choose', 0.8),
                (r'answer\s+to\s+each\s+question\s+will\s+be', 0.9),
                (r'using\s+the\s+mouse\s+and.*keypad', 0.9),
                (r'place\s+designated\s+to\s+enter', 0.85)
            ],
            'section_headers': [
                (r'^section[\s\-]*\d+$', 0.95),
                (r'^section[\s\-]*[a-z]$', 0.95),
                (r'^part[\s\-]*\d+$', 0.95),
                (r'^physics$', 0.8),
                (r'^chemistry$', 0.8),
                (r'^mathematics$', 0.8),
                (r'maximum\s+marks?\s*\d+', 0.85)
            ]
        }
    
    def is_instruction_box(self, text: str) -> Tuple[bool, float, str]:
        """
        Determine if the given text is an instruction box that should be filtered out
        Returns: (is_instruction, confidence, instruction_type)
        """
        if not text or len(text.strip()) < 10:
            return False, 0.0, ""
        
        text_lower = text.lower().strip()
        text_normalized = re.sub(r'\s+', ' ', text_lower)
        
        max_confidence = 0.0
        detected_type = ""
        
        # Check against all instruction patterns
        for instruction_type, patterns in self.instruction_patterns.items():
            for pattern, weight in patterns:
                try:
                    if re.search(pattern, text_normalized, re.IGNORECASE | re.MULTILINE):
                        confidence = weight
                        
                        # Boost confidence for multiple pattern matches
                        if max_confidence > 0:
                            confidence = min(0.98, confidence + 0.1)
                        
                        if confidence > max_confidence:
                            max_confidence = confidence
                            detected_type = instruction_type
                            
                except re.error as e:
                    logger.debug(f"Regex error with pattern {pattern}: {e}")
                    continue
        
        # Additional heuristics for instruction detection
        if max_confidence > 0.7:
            # Check for instruction-specific characteristics
            word_count = len(text_normalized.split())
            
            # Instructions tend to be longer and more formal
            if word_count > 50:
                max_confidence = min(0.98, max_confidence + 0.05)
            
            # Check for bullet points or numbered lists (common in instructions)
            if re.search(r'[•\*\-]\s+', text) or re.search(r'\d+\.\s+', text):
                max_confidence = min(0.98, max_confidence + 0.05)
            
            # Check for formal language patterns
            formal_patterns = [
                r'will\s+be\s+evaluated',
                r'according\s+to\s+the\s+following',
                r'each\s+question\s+has',
                r'only\s+one\s+of\s+these',
                r'if\s+none\s+of\s+the'
            ]
            
            formal_matches = sum(1 for pattern in formal_patterns 
                               if re.search(pattern, text_normalized))
            if formal_matches >= 2:
                max_confidence = min(0.98, max_confidence + 0.1)
        
        # Threshold for considering text as instruction
        is_instruction = max_confidence >= 0.75
        
        if is_instruction:
            logger.info(f"Detected instruction box (type: {detected_type}, confidence: {max_confidence:.2f})")
            logger.debug(f"Instruction text preview: {text[:100]}...")
        
        return is_instruction, max_confidence, detected_type
    
    def preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """Preprocess image for better OCR results with comprehensive error handling"""
        try:
            # Validate image path
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return None
                
            # Check file size
            file_size = os.path.getsize(image_path)
            if file_size > 20 * 1024 * 1024:  # 20MB limit
                logger.warning(f"Large image file: {file_size / 1024 / 1024:.1f}MB")
            
            # Load image with error handling
            img = cv2.imread(image_path)
            if img is None:
                # Try with PIL as fallback
                try:
                    pil_img = Image.open(image_path)
                    if pil_img.mode == 'RGBA':
                        # Convert RGBA to RGB
                        pil_img = pil_img.convert('RGB')
                    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    logger.error(f"Failed to read image {image_path}: {str(e)}")
                    return None
            
            # Validate image dimensions
            height, width = img.shape[:2]
            if height < 50 or width < 50:
                logger.warning(f"Image too small: {width}x{height}")
            elif height > 4000 or width > 4000:
                # Resize large images
                scale = min(4000/height, 4000/width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                logger.info(f"Resized large image from {width}x{height} to {new_width}x{new_height}")
            
            # Convert to grayscale
            try:
                if len(img.shape) == 2:
                    gray = img
                else:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                logger.error(f"Failed to convert to grayscale: {str(e)}")
                return None
            
            # Apply preprocessing with error handling
            try:
                # Apply gaussian blur to reduce noise
                blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            except Exception:
                # Fallback to no blur
                blurred = gray
                logger.warning("Gaussian blur failed, using original")
            
            try:
                # Apply adaptive thresholding
                thresh = cv2.adaptiveThreshold(
                    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 11, 2
                )
            except Exception as e:
                # Fallback to simple thresholding
                try:
                    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                except:
                    thresh = blurred
                logger.warning(f"Adaptive threshold failed, using fallback: {str(e)}")
            
            try:
                # Remove noise with morphological operations
                kernel = np.ones((1, 1), np.uint8)
                cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            except Exception:
                cleaned = thresh
                logger.warning("Morphological operations failed")
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Critical error in image preprocessing: {str(e)}\n{traceback.format_exc()}")
            return None
    
    def extract_text_ocr(self, image_path: str) -> str:
        """Extract text from image using OCR with comprehensive error handling"""
        try:
            # Validate input
            if not image_path or not os.path.exists(image_path):
                logger.error(f"Invalid image path for OCR: {image_path}")
                return ""
            
            # Try multiple preprocessing and OCR approaches
            best_text = ""
            best_confidence = 0
            
            # OCR configurations optimized for JEE questions
            configs = [
                r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz().,?!-+*=/ ',
                r'--oem 3 --psm 11',  # Sparse text
                r'--oem 3 --psm 3',   # Fully automatic
                r'--oem 1 --psm 6',   # Legacy engine
                r'--oem 3 --psm 12',  # Sparse text with OSD
            ]
            
            # Method 1: Direct OCR on original image
            for config in configs:
                try:
                    img = cv2.imread(image_path)
                    if img is None:
                        continue
                    
                    # Get OCR data with confidence scores
                    data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT, timeout=15)
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    
                    text = pytesseract.image_to_string(img, config=config, timeout=15)
                    
                    if text.strip() and avg_confidence > best_confidence:
                        best_text = text.strip()
                        best_confidence = avg_confidence
                        logger.debug(f"OCR confidence {avg_confidence:.1f}% with config: {config}")
                        
                except Exception as e:
                    logger.debug(f"OCR failed with config {config}: {str(e)}")
                    continue
            
            # Method 2: OCR on preprocessed image
            processed_img = self.preprocess_image(image_path)
            if processed_img is not None:
                for config in configs:
                    try:
                        data = pytesseract.image_to_data(processed_img, config=config, output_type=pytesseract.Output.DICT, timeout=15)
                        confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                        
                        text = pytesseract.image_to_string(processed_img, config=config, timeout=15)
                        
                        if text.strip() and avg_confidence > best_confidence:
                            best_text = text.strip()
                            best_confidence = avg_confidence
                            logger.debug(f"Preprocessed OCR confidence {avg_confidence:.1f}% with config: {config}")
                            
                    except Exception as e:
                        logger.debug(f"Preprocessed OCR failed: {str(e)}")
                        continue
            
            # Clean and enhance the best text result
            if best_text:
                # Fix common OCR errors in JEE questions
                replacements = [
                    (r'\bl\b', '1'),  # l to 1
                    (r'\bO\b', '0'),  # O to 0
                    (r'\|', 'I'),     # | to I
                    (r'0\)', 'D)'),   # 0) to D)
                    (r'8\)', 'B)'),   # 8) to B)
                    (r'C\)', 'C)'),   # Ensure proper formatting
                    (r'A\)', 'A)'),
                    (r'\s+', ' '),    # Normalize whitespace
                ]
                
                for pattern, replacement in replacements:
                    best_text = re.sub(pattern, replacement, best_text)
                
                # Remove excessive special characters but keep mathematical ones
                best_text = re.sub(r'[^\w\s\(\)\[\]\{\}.,;:!?+\-*/=<>%^&|~`"\'\n°∠∆∫∑∏√αβγδεζηθικλμνξοπρστυφχψω]', '', best_text)
                best_text = best_text.strip()
                
                logger.info(f"Best OCR result with {best_confidence:.1f}% confidence")
                return best_text
            
            logger.warning(f"No text extracted from {image_path}")
            return ""
            
        except Exception as e:
            logger.error(f"Critical error extracting text from {image_path}: {str(e)}\n{traceback.format_exc()}")
            return ""
    
    def analyze_question_structure(self, text: str) -> Dict:
        """Analyze text structure to identify question components"""
        text_lower = text.lower()
        
        # Count options
        option_patterns = [
            r'\([a-d]\)',  # (a), (b), (c), (d)
            r'[a-d][\.\)]',  # a. b. c. d.
            r'\([A-D]\)',  # (A), (B), (C), (D)
            r'[A-D][\.\)]'  # A. B. C. D.
        ]
        
        options_found = 0
        for pattern in option_patterns:
            matches = re.findall(pattern, text)
            options_found = max(options_found, len(matches))
        
        # Check for column structure
        has_columns = bool(re.search(r'column.*i.*column.*ii', text_lower))
        
        # Check for numerical answer requirements
        has_numerical = bool(re.search(r'nearest integer|integer value|answer.*\d+', text_lower))
        
        # Check for multiple correct indicators
        has_multiple = bool(re.search(r'one or more|multiple correct|more than one', text_lower))
        
        return {
            'options_count': options_found,
            'has_columns': has_columns,
            'has_numerical': has_numerical,
            'has_multiple': has_multiple,
            'text_length': len(text),
            'word_count': len(text.split())
        }
    
    def classify_question_type(self, text: str) -> Tuple[str, float]:
        """Classify question type based on text analysis with enhanced pattern matching"""
        text_lower = text.lower()
        structure = self.analyze_question_structure(text)
        
        # Enhanced scoring for each question type
        scores = {
            'SCQ': 0.0,
            'MCQ': 0.0,
            'Integer': 0.0,
            'MatchColumn': 0.0
        }
        
        # Match the Column detection - highest priority
        if structure['has_columns']:
            scores['MatchColumn'] += 0.9
            # Additional checks for match column
            if re.search(r'[A-D]\s*[-→]\s*[p-s]', text):  # A-p, B-q patterns
                scores['MatchColumn'] += 0.4
            if re.search(r'list\s*i.*list\s*ii', text_lower):
                scores['MatchColumn'] += 0.3
            
        for pattern, weight in self.patterns['match_column']['patterns']:
            if re.search(pattern, text_lower):
                scores['MatchColumn'] += weight * 0.3
        
        # Integer type detection - check for absence of options
        if structure['options_count'] < 4:  # No standard options
            scores['Integer'] += 0.5
            
            # Strong integer indicators
            if re.search(r'(answer|value)\s*(is|equals?\s*to)?\s*\d+', text_lower):
                scores['Integer'] += 0.6
            if re.search(r'find\s*(the\s*)?(value|answer)', text_lower):
                scores['Integer'] += 0.4
            if structure['has_numerical']:
                scores['Integer'] += 0.3
                
        for pattern, weight in self.patterns['integer']['patterns']:
            if re.search(pattern, text_lower):
                scores['Integer'] += weight * 0.2
        
        # MCQ detection - multiple correct answers
        if structure['has_multiple']:
            scores['MCQ'] += 0.85
            # Boost if we also see 4 options
            if structure['options_count'] == 4:
                scores['MCQ'] += 0.3
                
        for pattern, weight in self.patterns['mcq']['patterns']:
            if re.search(pattern, text_lower):
                scores['MCQ'] += weight * 0.3
        
        # Check for explicit MCQ instructions
        if re.search(r'select\s+(all|the)\s+correct', text_lower):
            scores['MCQ'] += 0.5
        
        # SCQ detection - single correct with 4 options
        if structure['options_count'] == 4 and not structure['has_multiple']:
            scores['SCQ'] += 0.7
            # Boost if no other strong indicators
            if not structure['has_columns'] and not structure['has_numerical']:
                scores['SCQ'] += 0.3
                
        for pattern, weight in self.patterns['scq']['patterns']:
            if re.search(pattern, text_lower):
                scores['SCQ'] += weight * 0.2
        
        # Additional SCQ indicators
        if re.search(r'which\s+(of\s+the\s+following|one)', text_lower):
            scores['SCQ'] += 0.3
        if re.search(r'correct\s+(option|answer|choice)\s+is', text_lower):
            scores['SCQ'] += 0.4
        
        # Apply penalties for conflicting indicators
        if structure['has_columns']:
            scores['SCQ'] *= 0.3
            scores['MCQ'] *= 0.3
            scores['Integer'] *= 0.2
        
        if structure['options_count'] == 4:
            scores['Integer'] *= 0.4  # Reduce integer score if options present
        
        # Normalize scores
        max_score = max(scores.values())
        if max_score > 0:
            for key in scores:
                scores[key] = scores[key] / max_score
        
        # Default fallbacks with confidence adjustment
        if max(scores.values()) < 0.3:
            if structure['options_count'] >= 4:
                scores['SCQ'] = 0.6  # Default to SCQ if options present
            else:
                scores['Integer'] = 0.5  # Default to Integer if no options
        
        # Get the type with highest score
        question_type = max(scores.items(), key=lambda x: x[1])
        confidence = min(question_type[1], 0.95)  # Cap confidence at 95%
        
        logger.debug(f"Question classification scores: {scores}")
        logger.info(f"Classified as {question_type[0]} with confidence {confidence:.2f}")
        
        return question_type[0], confidence
    
    def extract_options(self, text: str) -> List[str]:
        """Extract options from question text"""
        options = []
        
        # Try different option patterns
        patterns = [
            r'\([A-D]\)\s*([^(]+?)(?=\([A-D]\)|$)',  # (A) text (B) text
            r'[A-D][\.\)]\s*([^A-D]+?)(?=[A-D][\.\)]|$)',  # A. text B. text
            r'\([a-d]\)\s*([^(]+?)(?=\([a-d]\)|$)',  # (a) text (b) text
            r'[a-d][\.\)]\s*([^a-d]+?)(?=[a-d][\.\)]|$)'  # a. text b. text
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if len(matches) >= 3:  # At least 3 options found
                options = [opt.strip() for opt in matches]
                break
        
        return options[:4]  # Return maximum 4 options
    
    def analyze_question(self, image_path: str) -> Dict:
        """Complete question analysis with comprehensive error handling and instruction filtering"""
        try:
            # Extract text using OCR
            text = self.extract_text_ocr(image_path)
            
            if not text:
                return {
                    'question_type': 'Unknown',
                    'confidence': 0.0,
                    'ocr_text': '',
                    'options': [],
                    'structure': {},
                    'is_instruction': False
                }
            
            # Check if this is an instruction box that should be filtered out
            is_instruction, instruction_confidence, instruction_type = self.is_instruction_box(text)
            
            if is_instruction:
                logger.info(f"Filtering out instruction box: {instruction_type} (confidence: {instruction_confidence:.2f})")
                return {
                    'question_type': 'Instruction',
                    'confidence': instruction_confidence,
                    'ocr_text': text,
                    'options': [],
                    'structure': {
                        'instruction_type': instruction_type,
                        'filtered_out': True,
                        'reason': 'Detected as instruction box'
                    },
                    'is_instruction': True
                }
            
            # Classify question type for actual questions
            question_type, confidence = self.classify_question_type(text)
            
            # Extract options if applicable
            options = []
            if question_type in ['SCQ', 'MCQ']:
                options = self.extract_options(text)
            
            # Get structure analysis
            structure = self.analyze_question_structure(text)
            
            return {
                'question_type': question_type,
                'confidence': confidence,
                'ocr_text': text,
                'options': options,
                'structure': structure,
                'is_instruction': False
            }
            
        except Exception as e:
            logger.error(f"Error analyzing question {image_path}: {e}")
            return {
                'question_type': 'Unknown',
                'confidence': 0.0,
                'ocr_text': '',
                'options': [],
                'structure': {},
                'is_instruction': False
            }
    
    def batch_analyze(self, image_paths: List[str]) -> List[Dict]:
        """Analyze multiple questions in batch"""
        results = []
        
        for image_path in image_paths:
            result = self.analyze_question(image_path)
            result['image_path'] = image_path
            results.append(result)
        
        return results
    
    def get_analysis_summary(self, results: List[Dict]) -> Dict:
        """Get summary of analysis results"""
        if not results:
            return {}
        
        question_types = [r['question_type'] for r in results]
        type_counts = Counter(question_types)
        
        total_questions = len(results)
        avg_confidence = sum(r['confidence'] for r in results) / total_questions
        
        return {
            'total_questions': total_questions,
            'question_types': dict(type_counts),
            'average_confidence': avg_confidence,
            'high_confidence_count': sum(1 for r in results if r['confidence'] > 0.7),
            'uncertain_count': sum(1 for r in results if r['confidence'] < 0.3)
        }
