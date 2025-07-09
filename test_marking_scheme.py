#!/usr/bin/env python3
"""
Test script to verify the updated JEE marking scheme implementation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluator import TestEvaluator

def test_jee_advanced_mcq():
    """Test JEE Advanced MCQ marking scheme"""
    print("=== Testing JEE Advanced MCQ Marking Scheme ===")
    evaluator = TestEvaluator('jee_advanced')
    
    # Test case from marking scheme: correct answers are A, B, D
    correct_answer = ['A', 'B', 'D']
    
    test_cases = [
        # (user_answer, expected_marks, description)
        (['A', 'B', 'D'], 4, "All correct options chosen"),
        (['A', 'B'], 2, "2 correct options chosen (3+ correct available)"),
        (['A', 'D'], 2, "2 correct options chosen (3+ correct available)"),
        (['B', 'D'], 2, "2 correct options chosen (3+ correct available)"),
        (['A'], 1, "1 correct option chosen (2+ correct available)"),
        (['B'], 1, "1 correct option chosen (2+ correct available)"),
        (['D'], 1, "1 correct option chosen (2+ correct available)"),
        ([], 0, "No option chosen (unattempted)"),
        (['A', 'B', 'C'], -2, "2 correct + 1 wrong chosen"),
        (['A', 'C'], -2, "1 correct + 1 wrong chosen"),
        (['C'], -2, "Only wrong option chosen"),
        (['A', 'B', 'C', 'D'], -2, "All options chosen (including wrong)"),
    ]
    
    print(f"Correct Answer: {correct_answer}")
    print("-" * 60)
    
    for user_answer, expected_marks, description in test_cases:
        is_correct, marks = evaluator.evaluate_mcq(user_answer, correct_answer)
        status = "✓" if marks == expected_marks else "✗"
        print(f"{status} {user_answer:15} → {marks:3.0f} marks | {description}")
        if marks != expected_marks:
            print(f"   Expected: {expected_marks}, Got: {marks}")
    
    # Test case where all 4 options are correct but only 3 chosen
    print("\n--- Special Case: All 4 options correct, 3 chosen ---")
    correct_all_four = ['A', 'B', 'C', 'D']
    user_three = ['A', 'B', 'C']
    is_correct, marks = evaluator.evaluate_mcq(user_three, correct_all_four)
    expected = 3
    status = "✓" if marks == expected else "✗"
    print(f"{status} {user_three} → {marks} marks (Expected: {expected})")

def test_jee_main_marking():
    """Test JEE Main marking scheme"""
    print("\n=== Testing JEE Main Marking Scheme ===")
    evaluator = TestEvaluator('jee_main')
    
    # Test SCQ
    print("\n--- SCQ (Single Correct Questions) ---")
    scq_tests = [
        ('A', 'A', 4, "Correct answer"),
        ('B', 'A', -1, "Incorrect answer"),
        ('', 'A', 0, "Unattempted"),
        (None, 'A', 0, "Unattempted (None)"),
    ]
    
    for user_ans, correct_ans, expected, desc in scq_tests:
        is_correct, marks = evaluator.evaluate_scq(user_ans, correct_ans)
        status = "✓" if marks == expected else "✗"
        print(f"{status} '{user_ans}' vs '{correct_ans}' → {marks:3.0f} marks | {desc}")
    
    # Test Integer (Numerical Value Type)
    print("\n--- Integer (Numerical Value Type) ---")
    integer_tests = [
        (42, 42, 4, "Correct integer"),
        (41, 42, 0, "Incorrect integer (no negative marking)"),
        (None, 42, 0, "Unattempted"),
        (-5, 42, 0, "Negative number (invalid)"),
    ]
    
    for user_ans, correct_ans, expected, desc in integer_tests:
        is_correct, marks = evaluator.evaluate_integer(user_ans, correct_ans)
        status = "✓" if marks == expected else "✗"
        print(f"{status} {user_ans} vs {correct_ans} → {marks:3.0f} marks | {desc}")

def test_jee_advanced_other_sections():
    """Test other JEE Advanced sections"""
    print("\n=== Testing JEE Advanced Other Sections ===")
    evaluator = TestEvaluator('jee_advanced')
    
    # Test SCQ (Section 2)
    print("\n--- SCQ (Section 2: Single Correct Questions) ---")
    scq_tests = [
        ('A', 'A', 3, "Correct answer (+3 marks)"),
        ('B', 'A', -1, "Incorrect answer (-1 mark)"),
        ('', 'A', 0, "Unattempted"),
    ]
    
    for user_ans, correct_ans, expected, desc in scq_tests:
        is_correct, marks = evaluator.evaluate_scq(user_ans, correct_ans)
        status = "✓" if marks == expected else "✗"
        print(f"{status} '{user_ans}' vs '{correct_ans}' → {marks:3.0f} marks | {desc}")
    
    # Test Integer (Section 3)
    print("\n--- Integer (Section 3: Non-negative Integer) ---")
    integer_tests = [
        (15, 15, 4, "Correct integer (+4 marks)"),
        (14, 15, 0, "Incorrect integer (0 marks, no negative)"),
        (None, 15, 0, "Unattempted"),
    ]
    
    for user_ans, correct_ans, expected, desc in integer_tests:
        is_correct, marks = evaluator.evaluate_integer(user_ans, correct_ans)
        status = "✓" if marks == expected else "✗"
        print(f"{status} {user_ans} vs {correct_ans} → {marks:3.0f} marks | {desc}")

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n=== Testing Edge Cases ===")
    
    # Test both schemes
    for scheme in ['jee_main', 'jee_advanced']:
        print(f"\n--- {scheme.upper()} Edge Cases ---")
        evaluator = TestEvaluator(scheme)
        
        # Invalid inputs
        try:
            is_correct, marks = evaluator.evaluate_mcq(['X'], ['A'])
            print(f"✓ Invalid MCQ option handled: {marks} marks")
        except Exception as e:
            print(f"✗ Error with invalid MCQ option: {e}")
        
        try:
            is_correct, marks = evaluator.evaluate_integer("abc", 42)
            print(f"✓ Invalid integer handled: {marks} marks")
        except Exception as e:
            print(f"✗ Error with invalid integer: {e}")
        
        try:
            is_correct, marks = evaluator.evaluate_scq("X", "A")
            expected = evaluator.marking_rules['SCQ']['incorrect']
            status = "✓" if marks == expected else "✗"
            print(f"{status} Invalid SCQ option handled: {marks} marks")
        except Exception as e:
            print(f"✗ Error with invalid SCQ option: {e}")

def main():
    """Run all tests"""
    print("JEE Marking Scheme Implementation Test")
    print("=" * 50)
    
    test_jee_advanced_mcq()
    test_jee_main_marking()
    test_jee_advanced_other_sections()
    test_edge_cases()
    
    print("\n" + "=" * 50)
    print("Test completed. Check for any ✗ marks above.")

if __name__ == "__main__":
    main()