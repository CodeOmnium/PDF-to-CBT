import json
import logging
from typing import Dict, List, Any, Tuple, Optional
from models import Test, Question, Response, TestSession
from app import db
from sqlalchemy.exc import SQLAlchemyError
import traceback

logger = logging.getLogger(__name__)

class EvaluationError(Exception):
    """Custom exception for evaluation errors"""
    pass

class TestEvaluator:
    def __init__(self, marking_scheme: str = 'jee_main'):
        valid_schemes = ['jee_main', 'jee_advanced']
        if marking_scheme not in valid_schemes:
            logger.warning(f"Invalid marking scheme '{marking_scheme}', defaulting to 'jee_main'")
            marking_scheme = 'jee_main'
        self.marking_scheme = marking_scheme
        self.marking_rules = self.get_marking_rules()
    
    def get_marking_rules(self) -> Dict[str, Dict]:
        """Get marking rules based on scheme - Updated according to official marking scheme"""
        rules = {
            'jee_main': {
                # Section A: SCQ (Single Correct Questions)
                'SCQ': {'correct': 4, 'incorrect': -1, 'unattempted': 0},
                # Section B: Numerical Value Type (Integer)
                'Integer': {'correct': 4, 'incorrect': 0, 'unattempted': 0},  # No negative marking for numerical
                # Legacy support for MCQ and MatchColumn (not in current JEE Main)
                'MCQ': {'correct': 4, 'incorrect': -1, 'unattempted': 0},
                'MatchColumn': {'correct': 4, 'incorrect': -1, 'unattempted': 0}
            },
            'jee_advanced': {
                # Section 1: MCQ (Multiple Correct Questions) - 3 questions, 4 marks each
                'MCQ': {
                    'correct': 4,  # Full marks only if ALL correct options chosen
                    'partial_3_of_4': 3,  # If all 4 are correct but only 3 chosen
                    'partial_2_correct': 2,  # If 3+ correct but only 2 correct chosen
                    'partial_1_correct': 1,  # If 2+ correct but only 1 correct chosen
                    'incorrect': -2,  # All other cases
                    'unattempted': 0
                },
                # Section 2: SCQ (Single Correct Questions) - 4 questions, 3 marks each
                'SCQ': {'correct': 3, 'incorrect': -1, 'unattempted': 0},
                # Section 3: Integer Type - 6 questions, 4 marks each
                'Integer': {'correct': 4, 'incorrect': 0, 'unattempted': 0},  # No negative marking
                # Section 4: Match Column - 4 questions, 3 marks each
                'MatchColumn': {'correct': 3, 'incorrect': -1, 'unattempted': 0}
            }
        }
        
        return rules.get(self.marking_scheme, rules['jee_main'])
    
    def parse_answer(self, answer_str: str, question_type: str) -> Any:
        """Parse answer string based on question type with robust error handling"""
        if not answer_str or answer_str.strip() == '':
            return None
        
        try:
            if question_type == 'SCQ':
                answer = answer_str.upper().strip()
                # Validate SCQ answer (should be A, B, C, or D)
                if answer not in ['A', 'B', 'C', 'D']:
                    logger.warning(f"Invalid SCQ answer: {answer}")
                return answer
            
            elif question_type == 'MCQ':
                # Multiple answers as list
                if isinstance(answer_str, str):
                    answer_str = answer_str.strip()
                    if answer_str.startswith('[') and answer_str.endswith(']'):
                        try:
                            answers = json.loads(answer_str)
                            # Validate and clean
                            valid_answers = []
                            for ans in answers:
                                ans = str(ans).upper().strip()
                                if ans in ['A', 'B', 'C', 'D']:
                                    valid_answers.append(ans)
                            return sorted(list(set(valid_answers)))  # Remove duplicates and sort
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse MCQ JSON: {answer_str}")
                            return []
                    else:
                        # Single answer provided
                        answer = answer_str.upper().strip()
                        if answer in ['A', 'B', 'C', 'D']:
                            return [answer]
                        return []
                elif isinstance(answer_str, list):
                    # Already a list
                    valid_answers = []
                    for ans in answer_str:
                        ans = str(ans).upper().strip()
                        if ans in ['A', 'B', 'C', 'D']:
                            valid_answers.append(ans)
                    return sorted(list(set(valid_answers)))
                return []
            
            elif question_type == 'Integer':
                # Convert to integer with validation
                try:
                    # Handle various integer formats
                    answer_str = str(answer_str).strip()
                    # Remove common separators
                    answer_str = answer_str.replace(',', '').replace(' ', '')
                    value = float(answer_str)
                    int_value = int(value)
                    # Check if it's a whole number
                    if abs(value - int_value) < 0.0001:
                        # Validate range (typical JEE range)
                        if -999 <= int_value <= 999:
                            return int_value
                        else:
                            logger.warning(f"Integer answer out of range: {int_value}")
                            return int_value
                    else:
                        logger.warning(f"Non-integer value provided: {value}")
                        return int(round(value))
                except (ValueError, TypeError):
                    logger.error(f"Invalid integer answer: {answer_str}")
                    return None
            
            elif question_type == 'MatchColumn':
                # JEE Match Column: Single option (A, B, C, or D) representing a combination
                if isinstance(answer_str, str):
                    answer_str = answer_str.strip().upper()
                    
                    # For JEE format, it's a single option like 'A', 'B', 'C', or 'D'
                    if answer_str in ['A', 'B', 'C', 'D']:
                        return answer_str
                    
                    # Legacy support: Try to parse dictionary format
                    try:
                        if answer_str.startswith('{') and answer_str.endswith('}'):
                            matches = json.loads(answer_str)
                            # Validate match format
                            validated_matches = {}
                            for key, value in matches.items():
                                # Ensure keys and values are strings
                                key_str = str(key).upper().strip()
                                val_str = str(value).upper().strip()
                                # Validate JEE format: P,Q,R,S -> 1,2,3,4,5
                                if (key_str in ['P', 'Q', 'R', 'S'] and 
                                    val_str in ['1', '2', '3', '4', '5']):
                                    validated_matches[key_str] = val_str
                            return validated_matches if validated_matches else None
                        else:
                            # Try to parse simple format like "P-1, Q-2, R-3, S-4"
                            matches = {}
                            pairs = answer_str.split(',')
                            for pair in pairs:
                                if '-' in pair or '->' in pair:
                                    # Handle different separators
                                    separator = '-'
                                    if '->' in pair:
                                        separator = '->'
                                    
                                    key, val = pair.split(separator, 1)
                                    key = key.strip().upper()
                                    val = val.strip().upper()
                                    
                                    # Validate JEE format
                                    if (key in ['P', 'Q', 'R', 'S'] and 
                                        val in ['1', '2', '3', '4', '5']):
                                        matches[key] = val
                            return matches if matches else None
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.debug(f"Could not parse as dictionary, treating as single option: {answer_str}")
                        return None
                        
                elif isinstance(answer_str, dict):
                    # Already a dictionary - validate JEE format
                    validated_matches = {}
                    for key, value in answer_str.items():
                        key_str = str(key).upper().strip()
                        val_str = str(value).upper().strip()
                        if (key_str in ['P', 'Q', 'R', 'S'] and 
                            val_str in ['1', '2', '3', '4', '5']):
                            validated_matches[key_str] = val_str
                    return validated_matches if validated_matches else None
                    
                return None
            
        except Exception as e:
            logger.error(f"Unexpected error parsing answer {answer_str} for type {question_type}: {e}")
            return None
        
        return None
    
    def evaluate_scq(self, user_answer: str, correct_answer: str) -> Tuple[bool, float]:
        """Evaluate Single Correct Question with robust error handling"""
        try:
            # Handle empty or None answers
            if not user_answer or user_answer.strip() == '':
                return False, self.marking_rules['SCQ']['unattempted']
            
            if not correct_answer or correct_answer.strip() == '':
                logger.error("No correct answer provided for SCQ")
                return False, 0.0
            
            # Normalize answers
            user_ans = str(user_answer).upper().strip()
            correct_ans = str(correct_answer).upper().strip()
            
            # Validate answer format
            if user_ans not in ['A', 'B', 'C', 'D']:
                logger.warning(f"Invalid SCQ user answer format: {user_ans}")
                return False, self.marking_rules['SCQ']['incorrect']
            
            if correct_ans not in ['A', 'B', 'C', 'D']:
                logger.error(f"Invalid SCQ correct answer format: {correct_ans}")
                return False, 0.0
            
            is_correct = user_ans == correct_ans
            
            if is_correct:
                return True, float(self.marking_rules['SCQ']['correct'])
            else:
                return False, float(self.marking_rules['SCQ']['incorrect'])
                
        except Exception as e:
            logger.error(f"Error evaluating SCQ: {str(e)}")
            return False, 0.0
    
    def evaluate_mcq(self, user_answer: List[str], correct_answer: List[str]) -> Tuple[bool, float]:
        """Evaluate Multiple Correct Question according to official JEE marking schemes"""
        try:
            # Handle empty answers
            if not user_answer or (isinstance(user_answer, list) and len(user_answer) == 0):
                return False, float(self.marking_rules['MCQ']['unattempted'])
            
            if not correct_answer or (isinstance(correct_answer, list) and len(correct_answer) == 0):
                logger.error("No correct answer provided for MCQ")
                return False, 0.0
            
            # Ensure answers are lists
            if not isinstance(user_answer, list):
                user_answer = [str(user_answer)]
            if not isinstance(correct_answer, list):
                correct_answer = [str(correct_answer)]
            
            # Normalize and validate answers
            user_set = set()
            for ans in user_answer:
                ans_upper = str(ans).upper().strip()
                if ans_upper in ['A', 'B', 'C', 'D']:
                    user_set.add(ans_upper)
                else:
                    logger.warning(f"Invalid MCQ option in user answer: {ans}")
            
            correct_set = set()
            for ans in correct_answer:
                ans_upper = str(ans).upper().strip()
                if ans_upper in ['A', 'B', 'C', 'D']:
                    correct_set.add(ans_upper)
                else:
                    logger.error(f"Invalid MCQ option in correct answer: {ans}")
            
            if not correct_set:
                logger.error("No valid options in correct answer for MCQ")
                return False, 0.0
            
            # If no valid user answers
            if not user_set:
                return False, float(self.marking_rules['MCQ']['unattempted'])
            
            # Evaluate based on marking scheme
            if self.marking_scheme == 'jee_advanced':
                return self._evaluate_mcq_jee_advanced(user_set, correct_set)
            else:
                return self._evaluate_mcq_jee_main(user_set, correct_set)
                
        except Exception as e:
            logger.error(f"Error evaluating MCQ: {str(e)}\n{traceback.format_exc()}")
            return False, 0.0
    
    def _evaluate_mcq_jee_advanced(self, user_set: set, correct_set: set) -> Tuple[bool, float]:
        """Evaluate MCQ according to JEE Advanced marking scheme"""
        num_correct = len(correct_set)
        num_user_selected = len(user_set)
        num_correct_selected = len(user_set & correct_set)
        num_wrong_selected = len(user_set - correct_set)
        
        # JEE Advanced MCQ Marking Scheme:
        # Full Marks (+4): ONLY if ALL correct options are chosen
        if user_set == correct_set:
            return True, float(self.marking_rules['MCQ']['correct'])
        
        # Partial Marks: Only if NO wrong options are selected
        if num_wrong_selected == 0 and num_correct_selected > 0:
            # +3: If all 4 options are correct but only 3 options are chosen
            if num_correct == 4 and num_user_selected == 3:
                return False, float(self.marking_rules['MCQ']['partial_3_of_4'])
            
            # +2: If 3 or more options are correct but only 2 correct options are chosen
            elif num_correct >= 3 and num_user_selected == 2:
                return False, float(self.marking_rules['MCQ']['partial_2_correct'])
            
            # +1: If 2 or more options are correct but only 1 correct option is chosen
            elif num_correct >= 2 and num_user_selected == 1:
                return False, float(self.marking_rules['MCQ']['partial_1_correct'])
        
        # Negative Marks (-2): In all other cases (including wrong selections)
        return False, float(self.marking_rules['MCQ']['incorrect'])
    
    def _evaluate_mcq_jee_main(self, user_set: set, correct_set: set) -> Tuple[bool, float]:
        """Evaluate MCQ according to JEE Main marking scheme (legacy support)"""
        # For JEE Main, MCQs are typically treated as SCQs
        # But if MCQ marking is needed, use simple logic
        if user_set == correct_set:
            return True, float(self.marking_rules['MCQ']['correct'])
        else:
            return False, float(self.marking_rules['MCQ']['incorrect'])
    
    def evaluate_integer(self, user_answer: int, correct_answer: int) -> Tuple[bool, float]:
        """Evaluate Integer Type Question according to official marking schemes"""
        try:
            # Handle empty answer
            if user_answer is None:
                return False, float(self.marking_rules['Integer']['unattempted'])
            
            if correct_answer is None:
                logger.error("No correct answer provided for Integer question")
                return False, 0.0
            
            # Ensure both are integers
            try:
                user_ans = int(user_answer)
            except (ValueError, TypeError):
                logger.warning(f"Invalid integer user answer: {user_answer}")
                # For invalid format, treat as incorrect (no marks, no negative)
                return False, float(self.marking_rules['Integer']['incorrect'])
            
            try:
                correct_ans = int(correct_answer)
            except (ValueError, TypeError):
                logger.error(f"Invalid integer correct answer: {correct_answer}")
                return False, 0.0
            
            # Validate range (typical JEE range: 0-999 for non-negative integers)
            if user_ans < 0:
                logger.warning(f"User answer is negative: {user_ans} (JEE expects non-negative integers)")
                return False, float(self.marking_rules['Integer']['incorrect'])
            
            if not (0 <= user_ans <= 999):
                logger.warning(f"User answer out of typical range: {user_ans}")
            
            if not (0 <= correct_ans <= 999):
                logger.error(f"Correct answer out of typical range: {correct_ans}")
            
            is_correct = user_ans == correct_ans
            
            if is_correct:
                return True, float(self.marking_rules['Integer']['correct'])
            else:
                # Note: Both JEE Main and Advanced have 0 marks for incorrect integer answers
                return False, float(self.marking_rules['Integer']['incorrect'])
                
        except Exception as e:
            logger.error(f"Error evaluating Integer question: {str(e)}")
            return False, 0.0
    
    def evaluate_match_column(self, user_answer: Any, correct_answer: Any) -> Tuple[bool, float]:
        """Evaluate Match Column Question according to JEE format"""
        try:
            # Handle empty answer
            if not user_answer:
                return False, float(self.marking_rules['MatchColumn']['unattempted'])
            
            if not correct_answer:
                logger.error("No correct answer provided for Match Column question")
                return False, 0.0
            
            # JEE Match Column format: Single option (A, B, C, or D)
            if isinstance(user_answer, str) and isinstance(correct_answer, str):
                # Standard JEE format - single option selection
                user_ans = str(user_answer).upper().strip()
                correct_ans = str(correct_answer).upper().strip()
                
                # Validate format
                if user_ans not in ['A', 'B', 'C', 'D']:
                    logger.warning(f"Invalid Match Column user answer format: {user_ans}")
                    return False, float(self.marking_rules['MatchColumn']['incorrect'])
                
                if correct_ans not in ['A', 'B', 'C', 'D']:
                    logger.error(f"Invalid Match Column correct answer format: {correct_ans}")
                    return False, 0.0
                
                is_correct = user_ans == correct_ans
                
                if is_correct:
                    return True, float(self.marking_rules['MatchColumn']['correct'])
                else:
                    return False, float(self.marking_rules['MatchColumn']['incorrect'])
            
            # Legacy format: Dictionary matching (P->1, Q->2, etc.)
            elif isinstance(user_answer, dict) and isinstance(correct_answer, dict):
                # Normalize answers (uppercase keys and values)
                norm_user = {}
                norm_correct = {}
                
                for k, v in user_answer.items():
                    norm_user[str(k).upper().strip()] = str(v).upper().strip()
                
                for k, v in correct_answer.items():
                    norm_correct[str(k).upper().strip()] = str(v).upper().strip()
                
                if not norm_correct:
                    logger.error("Correct answer dictionary is empty")
                    return False, 0.0
                
                correct_matches = 0
                total_matches = len(norm_correct)
                has_wrong = False
                
                # Check each user answer
                for key, value in norm_user.items():
                    if key in norm_correct:
                        if norm_correct[key] == value:
                            correct_matches += 1
                        else:
                            has_wrong = True
                    else:
                        # User provided a key that doesn't exist in correct answer
                        has_wrong = True
                
                # Check for wrong answers - JEE Advanced has no partial marks for wrong selections
                if has_wrong:
                    return False, float(self.marking_rules['MatchColumn']['incorrect'])
                
                # Perfect match
                if correct_matches == total_matches and len(norm_user) == total_matches:
                    return True, float(self.marking_rules['MatchColumn']['correct'])
                
                # For JEE format, partial matching is not typically allowed
                # But if implemented, only give marks if no wrong answers
                if correct_matches > 0 and not has_wrong:
                    # In JEE Advanced, match column is typically all-or-nothing
                    return False, float(self.marking_rules['MatchColumn']['incorrect'])
                
                return False, float(self.marking_rules['MatchColumn']['unattempted'])
            
            else:
                logger.warning(f"Mismatched answer types: user={type(user_answer)}, correct={type(correct_answer)}")
                return False, float(self.marking_rules['MatchColumn']['incorrect'])
            
        except Exception as e:
            logger.error(f"Error evaluating Match Column: {str(e)}\n{traceback.format_exc()}")
            return False, 0.0
    
    def evaluate_response(self, response: Response) -> Dict:
        """Evaluate a single response"""
        question = response.question
        
        # Parse answers
        user_answer = self.parse_answer(response.user_answer, question.question_type)
        correct_answer = self.parse_answer(question.correct_answer, question.question_type)
        
        # Evaluate based on question type
        if question.question_type == 'SCQ':
            is_correct, marks = self.evaluate_scq(user_answer, correct_answer)
        elif question.question_type == 'MCQ':
            is_correct, marks = self.evaluate_mcq(user_answer, correct_answer)
        elif question.question_type == 'Integer':
            is_correct, marks = self.evaluate_integer(user_answer, correct_answer)
        elif question.question_type == 'MatchColumn':
            is_correct, marks = self.evaluate_match_column(user_answer, correct_answer)
        else:
            is_correct, marks = False, 0
        
        return {
            'is_correct': is_correct,
            'marks_awarded': marks,
            'question_type': question.question_type,
            'user_answer': user_answer,
            'correct_answer': correct_answer
        }
    
    def evaluate_test_session(self, session_id: int) -> Dict:
        """Evaluate complete test session with comprehensive error handling"""
        try:
            session = TestSession.query.get(session_id)
            if not session:
                raise EvaluationError(f"Test session {session_id} not found")
            
            responses = Response.query.filter_by(session_id=session_id).all()
            if not responses:
                logger.warning(f"No responses found for session {session_id}")
                return {
                    'session_id': session_id,
                    'total_questions': 0,
                    'attempted': 0,
                    'correct': 0,
                    'total_marks': 0,
                    'percentage': 0,
                    'question_wise_results': []
                }
            
            results = []
            total_marks = 0
            correct_count = 0
            attempted_count = 0
            
            for response in responses:
                try:
                    result = self.evaluate_response(response)
                    results.append({
                        'question_id': response.question_id,
                        'question_number': response.question.question_number,
                        'question_type': response.question.question_type,
                        'is_correct': result['is_correct'],
                        'marks_awarded': result['marks_awarded'],
                        'user_answer': result['user_answer'],
                        'correct_answer': result['correct_answer'],
                        'time_taken': response.time_taken
                    })
                    
                    total_marks += result['marks_awarded']
                    if result['is_correct']:
                        correct_count += 1
                    if result['user_answer'] is not None:
                        attempted_count += 1
                        
                except Exception as e:
                    logger.error(f"Error evaluating response {response.id}: {str(e)}")
                    # Add failed evaluation result
                    results.append({
                        'question_id': response.question_id,
                        'question_number': response.question.question_number,
                        'question_type': response.question.question_type,
                        'is_correct': False,
                        'marks_awarded': 0,
                        'user_answer': None,
                        'correct_answer': None,
                        'time_taken': response.time_taken,
                        'evaluation_error': str(e)
                    })
            
            # Calculate percentage
            max_possible_marks = len(responses) * max(
                self.marking_rules.get('SCQ', {}).get('correct', 4),
                self.marking_rules.get('MCQ', {}).get('correct', 4),
                self.marking_rules.get('Integer', {}).get('correct', 4),
                self.marking_rules.get('MatchColumn', {}).get('correct', 4)
            )
            
            percentage = (total_marks / max_possible_marks * 100) if max_possible_marks > 0 else 0
            
            # Update session with results
            try:
                session.total_marks = total_marks
                session.is_evaluated = True
                db.session.commit()
            except SQLAlchemyError as e:
                logger.error(f"Error updating session: {str(e)}")
                db.session.rollback()
            
            return {
                'session_id': session_id,
                'total_questions': len(responses),
                'attempted': attempted_count,
                'correct': correct_count,
                'total_marks': total_marks,
                'max_possible_marks': max_possible_marks,
                'percentage': percentage,
                'question_wise_results': results,
                'marking_scheme': self.marking_scheme
            }
            
        except Exception as e:
            logger.error(f"Critical error evaluating session {session_id}: {str(e)}\n{traceback.format_exc()}")
            raise EvaluationError(f"Failed to evaluate session: {str(e)}")