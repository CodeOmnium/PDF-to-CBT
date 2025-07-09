from flask import render_template, request, redirect, url_for, flash, jsonify, send_file, abort
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import os
import json
import logging
import gc
from datetime import datetime
from sqlalchemy.exc import SQLAlchemyError
from app import app, db
from models import Test, Question, TestSession, Response, AnswerKey
from pdf_processor import PDFProcessor, PDFProcessingError
from question_analyzer import QuestionAnalyzer
from evaluator import TestEvaluator
from utils import allowed_file, save_uploaded_file
import traceback

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def index():
    """Home page with error handling"""
    try:
        recent_tests = Test.query.order_by(Test.created_at.desc()).limit(5).all()
        return render_template('index.html', recent_tests=recent_tests)
    except SQLAlchemyError as e:
        logger.error(f"Database error on index page: {str(e)}")
        flash('Unable to load recent tests. Please try again later.', 'error')
        return render_template('index.html', recent_tests=[])

@app.route('/upload', methods=['GET', 'POST'])
def upload_pdf():
    """Upload PDF with comprehensive error handling"""
    if request.method == 'POST':
        try:
            # Validate file presence
            if 'pdf_file' not in request.files:
                flash('Please select a PDF file to upload', 'error')
                return redirect(request.url)
            
            file = request.files['pdf_file']
            
            # Validate file selection
            if file.filename == '':
                flash('Please select a file before clicking upload', 'error')
                return redirect(request.url)
            
            # Validate file type
            if not allowed_file(file.filename):
                flash('Invalid file type. Please upload a PDF file only.', 'error')
                return redirect(request.url)
            
            # Validate form data
            test_name = request.form.get('test_name', '').strip()
            if not test_name:
                flash('Test name is required', 'error')
                return redirect(request.url)
            
            if len(test_name) > 200:
                flash('Test name must be less than 200 characters', 'error')
                return redirect(request.url)
            
            # Check for duplicate test names
            existing_test = Test.query.filter_by(name=test_name).first()
            if existing_test:
                flash(f'A test with the name "{test_name}" already exists. Please choose a different name.', 'error')
                return redirect(request.url)
            
            marking_scheme = request.form.get('marking_scheme', 'jee_main')
            if marking_scheme not in ['jee_main', 'jee_advanced']:
                marking_scheme = 'jee_main'
            
            # Save uploaded file with error handling
            try:
                filename = save_uploaded_file(file)
                
                # Create test record
                test = Test(
                    name=test_name,
                    pdf_filename=filename,
                    marking_scheme=marking_scheme
                )
                db.session.add(test)
                db.session.commit()
                
                flash('PDF uploaded successfully!', 'success')
                return redirect(url_for('detect_boxes', test_id=test.id))
                
            except RequestEntityTooLarge:
                flash('File too large. Maximum file size is 50MB.', 'error')
                return redirect(request.url)
            except Exception as e:
                logger.error(f"Error saving file: {str(e)}\n{traceback.format_exc()}")
                flash('Error uploading file. Please try again.', 'error')
                return redirect(request.url)
                
        except SQLAlchemyError as e:
            db.session.rollback()
            logger.error(f"Database error during upload: {str(e)}")
            flash('Database error. Please try again.', 'error')
            return redirect(request.url)
        except Exception as e:
            logger.error(f"Unexpected error during upload: {str(e)}\n{traceback.format_exc()}")
            flash('An unexpected error occurred. Please try again.', 'error')
            return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/detect_boxes/<int:test_id>')
def detect_boxes(test_id):
    """Detect boxes in uploaded PDF with comprehensive error handling"""
    try:
        test = Test.query.get(test_id)
        if not test:
            flash('Test not found.', 'error')
            return redirect(url_for('all_tests'))
            
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], test.pdf_filename)
        
        # Verify PDF exists
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            flash('PDF file not found. Please upload again.', 'error')
            return redirect(url_for('upload_pdf'))
        
        processor = None
        try:
            # Initialize enhanced PDF processor
            processor = PDFProcessor(pdf_path)
            
            # Detect boxes with enhanced methods
            boxes = processor.detect_all_boxes()
            
            # Get processing statistics
            stats = processor.get_processing_stats()
            logger.info(f"PDF processing completed: {stats}")
            
            if not boxes:
                flash('No boxes detected in the PDF. Please ensure the PDF contains pre-drawn rectangular boxes around questions.', 'warning')
                if processor:
                    processor.cleanup()
                return render_template('box_detection.html', test=test, boxes=[], analysis_results=[])
            
            # Log detection summary
            logger.info(f"Detected {len(boxes)} boxes using methods: {', '.join(stats.get('methods_used', []))}")
            
            # Create images directory for this test
            images_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'images', str(test_id))
            os.makedirs(images_dir, exist_ok=True)
            
            # Extract question images with enhanced processing
            extracted_questions = processor.extract_all_questions_enhanced(images_dir)
            
            if not extracted_questions:
                flash('No questions could be extracted from the detected boxes.', 'warning')
                return render_template('box_detection.html', test=test, boxes=boxes, analysis_results=[])
            
            # Delete existing questions for this test (for re-processing)
            Question.query.filter_by(test_id=test_id).delete()
            
            # Analyze questions with error handling
            analyzer = QuestionAnalyzer()
            analysis_results = []
            successful_questions = 0
            
            for question_data in extracted_questions:
                try:
                    analysis = analyzer.analyze_question(question_data['image_path'])
                    analysis_results.append(analysis)
                    
                    # Validate analysis results
                    if analysis.get('question_type') not in ['SCQ', 'MCQ', 'Integer', 'MatchColumn']:
                        analysis['question_type'] = 'SCQ'  # Default fallback
                    
                    # Create question record with validation
                    question = Question(
                        test_id=test_id,
                        question_number=question_data['question_number'],
                        question_type=analysis['question_type'],
                        image_path=os.path.relpath(question_data['image_path'], app.config['UPLOAD_FOLDER']),
                        ocr_text=analysis.get('ocr_text', '')[:5000],  # Limit text length
                        confidence_score=max(0.0, min(1.0, analysis.get('confidence', 0.5))),  # Ensure valid range
                        box_coordinates=json.dumps(question_data['coordinates']),
                        page_number=question_data['page_number']
                    )
                    db.session.add(question)
                    successful_questions += 1
                    
                except Exception as e:
                    logger.error(f"Error analyzing question {question_data['question_number']}: {str(e)}")
                    analysis_results.append({
                        'question_type': 'Unknown',
                        'confidence': 0.0,
                        'ocr_text': 'Error analyzing question',
                        'error': str(e)
                    })
            
            # Update test with transaction protection
            try:
                test.total_questions = successful_questions
                db.session.commit()
                
                if successful_questions < len(extracted_questions):
                    flash(f'Detected {len(boxes)} boxes. Successfully processed {successful_questions} out of {len(extracted_questions)} questions.', 'warning')
                else:
                    flash(f'Successfully detected {len(boxes)} boxes and extracted {successful_questions} questions!', 'success')
                    
            except SQLAlchemyError as e:
                db.session.rollback()
                logger.error(f"Database error saving questions: {str(e)}")
                flash('Error saving questions to database. Please try again.', 'error')
                
            return render_template('box_detection.html', test=test, boxes=boxes, analysis_results=analysis_results)
            
        except PDFProcessingError as e:
            logger.error(f"PDF processing error: {str(e)}")
            flash(f'PDF processing error: {str(e)}', 'error')
            return redirect(url_for('upload_pdf'))
        finally:
            if processor:
                try:
                    processor.close()
                except:
                    pass
                    
    except Exception as e:
        logger.error(f"Unexpected error in detect_boxes: {str(e)}\n{traceback.format_exc()}")
        flash('An unexpected error occurred while processing the PDF.', 'error')
        return redirect(url_for('all_tests'))

@app.route('/test/<int:test_id>')
def test_interface(test_id):
    """Test interface for students"""
    test = Test.query.get_or_404(test_id)
    return render_template('test_interface.html', test=test)

@app.route('/start_test/<int:test_id>', methods=['POST'])
def start_test(test_id):
    """Start a new test session"""
    test = Test.query.get_or_404(test_id)
    student_name = request.form.get('student_name', '').strip()
    
    if not student_name:
        flash('Student name is required', 'error')
        return redirect(url_for('test_interface', test_id=test_id))
    
    # Create test session
    session = TestSession(
        test_id=test_id,
        student_name=student_name
    )
    db.session.add(session)
    db.session.commit()
    
    return redirect(url_for('take_test', session_id=session.id))

@app.route('/take_test/<int:session_id>')
def take_test(session_id):
    """Take test interface"""
    session = TestSession.query.get_or_404(session_id)
    questions = Question.query.filter_by(test_id=session.test_id).order_by(Question.question_number).all()
    
    return render_template('test_interface.html', session=session, questions=questions)

@app.route('/api/save_answer', methods=['POST'])
def save_answer():
    """Save answer via API"""
    data = request.json
    session_id = data.get('session_id')
    question_id = data.get('question_id')
    answer = data.get('answer')
    time_spent = data.get('time_spent', 0)
    
    try:
        # Find or create response
        response = Response.query.filter_by(
            test_session_id=session_id,
            question_id=question_id
        ).first()
        
        if response:
            response.user_answer = json.dumps(answer) if isinstance(answer, (list, dict)) else str(answer)
            response.time_spent = time_spent
        else:
            response = Response(
                test_session_id=session_id,
                question_id=question_id,
                user_answer=json.dumps(answer) if isinstance(answer, (list, dict)) else str(answer),
                time_spent=time_spent
            )
            db.session.add(response)
        
        db.session.commit()
        
        return jsonify({'status': 'success'})
        
    except Exception as e:
        logger.error(f"Error saving answer: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/submit_test/<int:session_id>', methods=['POST'])
def submit_test(session_id):
    """Submit test and show results"""
    session = TestSession.query.get_or_404(session_id)
    session.end_time = datetime.utcnow()
    db.session.commit()
    
    # Evaluate test
    evaluator = TestEvaluator(session.test.marking_scheme)
    results = evaluator.generate_detailed_report(session_id)
    
    return render_template('results.html', session=session, results=results)

@app.route('/answer_key/<int:test_id>', methods=['GET', 'POST'])
def answer_key(test_id):
    """Upload and manage answer key"""
    test = Test.query.get_or_404(test_id)
    
    if request.method == 'POST':
        # Handle manual answer key input
        answers = {}
        questions = Question.query.filter_by(test_id=test_id).all()
        
        for question in questions:
            answer_key = f'answer_{question.id}'
            if answer_key in request.form:
                answer_value = request.form[answer_key]
                
                # Parse answer based on question type
                if question.question_type == 'MCQ':
                    # Multiple answers
                    answer_value = request.form.getlist(answer_key)
                elif question.question_type == 'MatchColumn':
                    # Match column answers
                    match_answers = {}
                    for i in ['A', 'B', 'C', 'D']:
                        match_key = f'match_{question.id}_{i}'
                        if match_key in request.form:
                            match_answers[i] = request.form[match_key]
                    answer_value = match_answers
                
                answers[question.id] = answer_value
                
                # Update question
                question.correct_answer = json.dumps(answer_value) if isinstance(answer_value, (list, dict)) else str(answer_value)
        
        # Save answer key
        answer_key_obj = AnswerKey(
            test_id=test_id,
            answers=json.dumps(answers)
        )
        db.session.add(answer_key_obj)
        db.session.commit()
        
        flash('Answer key saved successfully!', 'success')
        return redirect(url_for('test_interface', test_id=test_id))
    
    questions = Question.query.filter_by(test_id=test_id).order_by(Question.question_number).all()
    return render_template('answer_key.html', test=test, questions=questions)

@app.route('/api/get_question_image/<int:question_id>')
def get_question_image(question_id):
    """Get question image"""
    question = Question.query.get_or_404(question_id)
    try:
        return send_file(question.image_path, mimetype='image/png')
    except Exception as e:
        logger.error(f"Error serving image: {e}")
        return "Image not found", 404

@app.route('/results/<int:session_id>')
def view_results(session_id):
    """View test results"""
    session = TestSession.query.get_or_404(session_id)
    
    if session.status != 'completed':
        flash('Test not completed yet', 'warning')
        return redirect(url_for('take_test', session_id=session_id))
    
    evaluator = TestEvaluator(session.test.marking_scheme)
    results = evaluator.generate_detailed_report(session_id)
    
    return render_template('results.html', session=session, results=results)

@app.route('/all_tests')
def all_tests():
    """View all tests"""
    tests = Test.query.order_by(Test.created_at.desc()).all()
    return render_template('all_tests.html', tests=tests)

@app.route('/test_sessions/<int:test_id>')
def test_sessions(test_id):
    """View all sessions for a test"""
    test = Test.query.get_or_404(test_id)
    sessions = TestSession.query.filter_by(test_id=test_id).order_by(TestSession.start_time.desc()).all()
    return render_template('test_sessions.html', test=test, sessions=sessions)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    logger.error(f"Internal server error: {e}")
    return render_template('500.html'), 500
