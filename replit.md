# PDF Question Extraction Tool

## Overview

This is a Flask-based web application designed to process PDF files containing pre-drawn boxes around questions, automatically extract these questions as images, analyze their types, and create interactive test interfaces. The tool is specifically designed for JEE Main/Advanced exam preparation, supporting various question types including Single Correct Questions (SCQ), Multiple Correct Questions (MCQ), Integer type questions, and Match the Column questions.

## System Architecture

### Backend Architecture
- **Framework**: Flask with SQLAlchemy ORM
- **Database**: SQLite (configurable to PostgreSQL via environment variable)
- **File Processing**: PyMuPDF (fitz) for PDF processing, OpenCV for image processing
- **OCR**: Tesseract via pytesseract for text extraction
- **NLP**: spaCy for question type analysis

### Frontend Architecture
- **UI Framework**: Bootstrap 5 with dark theme
- **JavaScript Libraries**: PDF.js for PDF rendering, custom JavaScript for interactive features
- **Templates**: Jinja2 templating with responsive design
- **Icons**: Font Awesome for consistent iconography

### Key Components

1. **PDF Processing Engine** (`pdf_processor.py`)
   - Detects rectangular annotations and shapes in PDF files
   - Extracts question regions as high-quality images
   - Supports multiple box detection methods (annotations, OpenCV shape detection)

2. **Question Analysis Engine** (`question_analyzer.py`)
   - Analyzes extracted question images to determine question type
   - Uses OCR and NLP to identify question patterns
   - Supports SCQ, MCQ, Integer, and Match Column question types

3. **Test Evaluation System** (`evaluator.py`)
   - Implements JEE Main/Advanced marking schemes
   - Handles different scoring rules for each question type
   - Provides detailed performance analytics

4. **Web Interface** (`routes.py`)
   - Upload and process PDF files
   - Interactive test-taking interface
   - Answer key management
   - Results and analytics dashboard

## Data Flow

1. **PDF Upload**: User uploads PDF with pre-drawn boxes around questions
2. **Box Detection**: System automatically detects rectangular annotations/shapes
3. **Image Extraction**: Questions are cropped and extracted as individual images
4. **Question Analysis**: Each question is analyzed to determine its type and structure
5. **Test Creation**: Interactive test interface is generated with navigation and timing
6. **Answer Collection**: User responses are collected and stored during test sessions
7. **Evaluation**: Responses are evaluated against answer keys using appropriate marking schemes
8. **Results**: Detailed performance reports are generated with analytics

## Database Schema

### Tables
- **Test**: Stores test metadata (name, PDF filename, marking scheme, creation date)
- **Question**: Stores individual question data (image path, type, coordinates, OCR text)
- **TestSession**: Tracks student test sessions (name, timing, scores, status)
- **Response**: Stores student answers for each question in a session

### Relationships
- Test → Questions (One-to-Many)
- Test → TestSessions (One-to-Many)
- TestSession → Responses (One-to-Many)
- Question → Responses (One-to-Many)

## External Dependencies

### Python Packages
- Flask, Flask-SQLAlchemy for web framework and database ORM
- PyMuPDF (fitz) for PDF processing
- OpenCV for computer vision and image processing
- Pillow for image manipulation
- pytesseract for OCR text extraction
- spaCy for natural language processing
- Werkzeug for file handling utilities

### Frontend Libraries
- Bootstrap 5 for responsive UI components
- PDF.js for client-side PDF rendering
- Font Awesome for icons
- Custom JavaScript for interactive features

### System Dependencies
- Tesseract OCR engine for text recognition
- spaCy English language model (en_core_web_sm)

## Configuration

### Environment Variables
- `SESSION_SECRET`: Flask session secret key
- `DATABASE_URL`: Database connection string (defaults to SQLite)
- File upload settings: 50MB max file size, dedicated upload directories

### Upload Configuration
- Supported file types: PDF only
- Upload directory: `uploads/` with subdirectory for extracted images
- Unique filename generation to prevent conflicts

## Deployment Strategy

The application is designed for deployment on Replit with the following considerations:

1. **Database**: Uses SQLite by default, easily configurable for PostgreSQL
2. **File Storage**: Local file system for uploaded PDFs and extracted images
3. **Environment**: Configured for development with debug mode enabled
4. **Proxy Support**: Includes ProxyFix middleware for deployment behind reverse proxies
5. **Static Assets**: Served via Flask's static file handling

## User Preferences

Preferred communication style: Simple, everyday language.

## Changelog

Changelog:
- July 07, 2025. Initial setup
- July 07, 2025. Enhanced PDF processing with multiple detection methods for improved box detection accuracy
- July 07, 2025. Improved OCR engine with confidence scoring and multiple preprocessing approaches
- July 07, 2025. Enhanced question type classification with better pattern matching and scoring
- July 07, 2025. Added custom CSS for improved user interface and animations
- July 07, 2025. Completed migration from Replit Agent to Replit environment with all security best practices