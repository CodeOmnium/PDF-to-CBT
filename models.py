from app import db
from datetime import datetime
from sqlalchemy import Text, JSON, event, CheckConstraint, Index
from sqlalchemy.orm import validates
import json
import re

class Test(db.Model):
    __tablename__ = 'test'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False, unique=True)
    pdf_filename = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    marking_scheme = db.Column(db.String(20), default='jee_main', nullable=False)
    total_questions = db.Column(db.Integer, default=0, nullable=False)
    
    # Relationships
    questions = db.relationship('Question', backref='test', lazy=True, cascade='all, delete-orphan')
    test_sessions = db.relationship('TestSession', backref='test', lazy=True, cascade='all, delete-orphan')
    
    # Add constraints
    __table_args__ = (
        CheckConstraint('LENGTH(name) >= 1 AND LENGTH(name) <= 200', name='check_test_name_length'),
        CheckConstraint("marking_scheme IN ('jee_main', 'jee_advanced')", name='check_marking_scheme'),
        CheckConstraint('total_questions >= 0 AND total_questions <= 300', name='check_total_questions_range'),
        Index('idx_test_created', 'created_at'),
        Index('idx_test_name', 'name'),
    )
    
    @validates('name')
    def validate_name(self, key, name):
        if not name or not name.strip():
            raise ValueError("Test name cannot be empty")
        name = name.strip()
        if len(name) > 200:
            raise ValueError("Test name must be less than 200 characters")
        # Sanitize name
        name = re.sub(r'[<>&"\']', '', name)
        return name
    
    @validates('pdf_filename')
    def validate_pdf_filename(self, key, filename):
        if not filename:
            raise ValueError("PDF filename cannot be empty")
        if not filename.lower().endswith('.pdf'):
            raise ValueError("File must be a PDF")
        return filename
    
    @validates('marking_scheme')
    def validate_marking_scheme(self, key, scheme):
        valid_schemes = ['jee_main', 'jee_advanced']
        if scheme not in valid_schemes:
            raise ValueError(f"Invalid marking scheme. Must be one of: {', '.join(valid_schemes)}")
        return scheme
    
    @validates('total_questions')
    def validate_total_questions(self, key, value):
        if value < 0:
            raise ValueError("Total questions cannot be negative")
        if value > 300:
            raise ValueError("Total questions cannot exceed 300")
        return value

class Question(db.Model):
    __tablename__ = 'question'
    
    id = db.Column(db.Integer, primary_key=True)
    test_id = db.Column(db.Integer, db.ForeignKey('test.id'), nullable=False)
    question_number = db.Column(db.Integer, nullable=False)
    question_type = db.Column(db.String(20), nullable=False)
    image_path = db.Column(db.String(500), nullable=False)
    ocr_text = db.Column(Text)
    confidence_score = db.Column(db.Float, default=0.0, nullable=False)
    correct_answer = db.Column(Text)  # JSON string for complex answers
    box_coordinates = db.Column(Text)  # JSON string for coordinates
    page_number = db.Column(db.Integer, default=1, nullable=False)
    
    # Relationships
    responses = db.relationship('Response', backref='question', lazy=True, cascade='all, delete-orphan')
    
    # Add constraints
    __table_args__ = (
        CheckConstraint("question_type IN ('SCQ', 'MCQ', 'Integer', 'MatchColumn')", name='check_question_type'),
        CheckConstraint('question_number > 0 AND question_number <= 300', name='check_question_number_range'),
        CheckConstraint('confidence_score >= 0.0 AND confidence_score <= 1.0', name='check_confidence_score_range'),
        CheckConstraint('page_number > 0', name='check_page_number_positive'),
        Index('idx_question_test', 'test_id'),
        Index('idx_question_number', 'test_id', 'question_number'),
    )
    
    @validates('question_number')
    def validate_question_number(self, key, value):
        if value <= 0:
            raise ValueError("Question number must be positive")
        if value > 300:
            raise ValueError("Question number cannot exceed 300")
        return value
    
    @validates('question_type')
    def validate_question_type(self, key, q_type):
        valid_types = ['SCQ', 'MCQ', 'Integer', 'MatchColumn']
        if q_type not in valid_types:
            raise ValueError(f"Invalid question type. Must be one of: {', '.join(valid_types)}")
        return q_type
    
    @validates('confidence_score')
    def validate_confidence_score(self, key, score):
        if score < 0.0 or score > 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")
        return round(score, 4)  # Round to 4 decimal places
    
    @validates('page_number')
    def validate_page_number(self, key, value):
        if value <= 0:
            raise ValueError("Page number must be positive")
        return value
    
    @validates('correct_answer')
    def validate_correct_answer(self, key, answer):
        if answer:
            try:
                # Validate JSON format if provided
                json.loads(answer)
            except json.JSONDecodeError:
                # If not JSON, it's a simple answer - that's fine
                pass
        return answer
    
    @validates('box_coordinates')
    def validate_box_coordinates(self, key, coords):
        if coords:
            try:
                # Validate JSON format
                data = json.loads(coords)
                if not isinstance(data, list) or len(data) != 4:
                    raise ValueError("Box coordinates must be a list of 4 numbers")
            except json.JSONDecodeError:
                raise ValueError("Box coordinates must be valid JSON")
        return coords

class TestSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    test_id = db.Column(db.Integer, db.ForeignKey('test.id'), nullable=False)
    student_name = db.Column(db.String(100), nullable=False)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime)
    total_score = db.Column(db.Float, default=0.0)
    total_marks = db.Column(db.Float, default=0.0)
    status = db.Column(db.String(20), default='active')  # active, completed
    
    # Relationships
    responses = db.relationship('Response', backref='test_session', lazy=True, cascade='all, delete-orphan')

class Response(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    test_session_id = db.Column(db.Integer, db.ForeignKey('test_session.id'), nullable=False)
    question_id = db.Column(db.Integer, db.ForeignKey('question.id'), nullable=False)
    user_answer = db.Column(db.String(200))  # JSON string for complex answers
    is_correct = db.Column(db.Boolean)
    marks_awarded = db.Column(db.Float, default=0.0)
    time_spent = db.Column(db.Integer, default=0)  # seconds
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)

class AnswerKey(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    test_id = db.Column(db.Integer, db.ForeignKey('test.id'), nullable=False)
    answers = db.Column(Text)  # JSON string containing all answers
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    test = db.relationship('Test', backref='answer_keys', lazy=True)
