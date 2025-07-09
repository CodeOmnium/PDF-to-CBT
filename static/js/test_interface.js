/**
 * Test Interface JavaScript
 * Handles test-taking functionality, navigation, timer, and answer saving
 */

class TestInterface {
    constructor(options) {
        this.sessionId = options.sessionId;
        this.questions = options.questions;
        this.testId = options.testId;
        this.currentQuestion = 1;
        this.answers = new Map();
        this.timeSpent = new Map();
        this.startTime = Date.now();
        this.questionStartTime = Date.now();
        this.timerInterval = null;
        
        this.init();
    }
    
    init() {
        this.setupTimer();
        this.setupNavigation();
        this.setupKeyboardShortcuts();
        this.loadQuestion(1);
        this.startAutoSave();
        
        // Initialize time tracking for each question
        this.questions.forEach((q, index) => {
            this.timeSpent.set(index + 1, 0);
        });
    }
    
    setupTimer() {
        this.timerInterval = setInterval(() => {
            this.updateTimer();
        }, 1000);
    }
    
    updateTimer() {
        const elapsed = Date.now() - this.startTime;
        const hours = Math.floor(elapsed / 3600000);
        const minutes = Math.floor((elapsed % 3600000) / 60000);
        const seconds = Math.floor((elapsed % 60000) / 1000);
        
        const timerElement = document.getElementById('timer');
        if (timerElement) {
            timerElement.textContent = 
                `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }
    }
    
    setupNavigation() {
        const prevBtn = document.getElementById('prevBtn');
        const nextBtn = document.getElementById('nextBtn');
        
        if (prevBtn) {
            prevBtn.addEventListener('click', () => this.previousQuestion());
        }
        
        if (nextBtn) {
            nextBtn.addEventListener('click', () => this.nextQuestion());
        }
        
        this.updateNavigationButtons();
    }
    
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Only handle shortcuts when not in input fields
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') {
                return;
            }
            
            switch(e.key) {
                case 'ArrowLeft':
                    e.preventDefault();
                    this.previousQuestion();
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    this.nextQuestion();
                    break;
                case '1':
                case '2':
                case '3':
                case '4':
                    e.preventDefault();
                    this.selectOption(e.key);
                    break;
            }
        });
    }
    
    selectOption(optionNumber) {
        const question = this.questions[this.currentQuestion - 1];
        if (question && question.question_type === 'SCQ') {
            const options = ['A', 'B', 'C', 'D'];
            const optionValue = options[parseInt(optionNumber) - 1];
            if (optionValue) {
                const radio = document.querySelector(`input[name="answer"][value="${optionValue}"]`);
                if (radio) {
                    radio.checked = true;
                    this.saveAnswer(optionValue);
                }
            }
        }
    }
    
    loadQuestion(questionNumber) {
        const question = this.questions[questionNumber - 1];
        if (!question) return;
        
        this.trackQuestionTime();
        this.currentQuestion = questionNumber;
        this.questionStartTime = Date.now();
        
        const container = document.getElementById('questionContainer');
        if (!container) return;
        
        container.innerHTML = this.renderQuestion(question);
        this.updateNavigationButtons();
        this.updateQuestionNavigation();
        this.updateProgress();
        
        // Load saved answer if exists
        const savedAnswer = this.answers.get(questionNumber);
        if (savedAnswer) {
            this.loadSavedAnswer(question, savedAnswer);
        }
        
        // Add fade-in animation
        container.classList.add('fade-in');
        setTimeout(() => container.classList.remove('fade-in'), 300);
    }
    
    renderQuestion(question) {
        const imageUrl = `/api/get_question_image/${question.id}`;
        
        let answerInterface = '';
        
        switch(question.question_type) {
            case 'SCQ':
                answerInterface = this.renderSCQInterface();
                break;
            case 'MCQ':
                answerInterface = this.renderMCQInterface();
                break;
            case 'Integer':
                answerInterface = this.renderIntegerInterface();
                break;
            case 'MatchColumn':
                answerInterface = this.renderMatchColumnInterface();
                break;
            default:
                answerInterface = '<p class="text-muted">Unknown question type</p>';
        }
        
        return `
            <div class="row">
                <div class="col-md-8">
                    <div class="question-image-container mb-3">
                        <img src="${imageUrl}" class="question-image img-fluid" alt="Question ${question.question_number}">
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="question-info mb-3">
                        <h5>Question ${question.question_number}</h5>
                        <span class="badge bg-${this.getQuestionTypeBadgeColor(question.question_type)}">${question.question_type}</span>
                        <span class="badge bg-secondary ms-2">Page ${question.page_number}</span>
                    </div>
                    <div class="answer-section">
                        ${answerInterface}
                    </div>
                    <div class="question-actions mt-3">
                        <button class="btn btn-sm btn-outline-warning" onclick="testInterface.flagQuestion(${question.question_number})">
                            <i class="fas fa-flag"></i> Flag for Review
                        </button>
                        <button class="btn btn-sm btn-outline-secondary" onclick="testInterface.clearAnswer()">
                            <i class="fas fa-eraser"></i> Clear Answer
                        </button>
                    </div>
                </div>
            </div>
        `;
    }
    
    renderSCQInterface() {
        return `
            <div class="answer-options">
                <h6>Select the correct answer:</h6>
                <div class="form-check mb-2">
                    <input class="form-check-input" type="radio" name="answer" value="A" id="optionA" onchange="testInterface.saveAnswer('A')">
                    <label class="form-check-label" for="optionA">A</label>
                </div>
                <div class="form-check mb-2">
                    <input class="form-check-input" type="radio" name="answer" value="B" id="optionB" onchange="testInterface.saveAnswer('B')">
                    <label class="form-check-label" for="optionB">B</label>
                </div>
                <div class="form-check mb-2">
                    <input class="form-check-input" type="radio" name="answer" value="C" id="optionC" onchange="testInterface.saveAnswer('C')">
                    <label class="form-check-label" for="optionC">C</label>
                </div>
                <div class="form-check mb-2">
                    <input class="form-check-input" type="radio" name="answer" value="D" id="optionD" onchange="testInterface.saveAnswer('D')">
                    <label class="form-check-label" for="optionD">D</label>
                </div>
            </div>
        `;
    }
    
    renderMCQInterface() {
        return `
            <div class="answer-options">
                <h6>Select all correct answers:</h6>
                <div class="form-check mb-2">
                    <input class="form-check-input" type="checkbox" name="answer" value="A" id="optionA" onchange="testInterface.saveMCQAnswer()">
                    <label class="form-check-label" for="optionA">A</label>
                </div>
                <div class="form-check mb-2">
                    <input class="form-check-input" type="checkbox" name="answer" value="B" id="optionB" onchange="testInterface.saveMCQAnswer()">
                    <label class="form-check-label" for="optionB">B</label>
                </div>
                <div class="form-check mb-2">
                    <input class="form-check-input" type="checkbox" name="answer" value="C" id="optionC" onchange="testInterface.saveMCQAnswer()">
                    <label class="form-check-label" for="optionC">C</label>
                </div>
                <div class="form-check mb-2">
                    <input class="form-check-input" type="checkbox" name="answer" value="D" id="optionD" onchange="testInterface.saveMCQAnswer()">
                    <label class="form-check-label" for="optionD">D</label>
                </div>
            </div>
        `;
    }
    
    renderIntegerInterface() {
        return `
            <div class="answer-options">
                <h6>Enter your answer:</h6>
                <div class="input-group">
                    <input type="number" class="form-control" id="integerAnswer" placeholder="Enter integer (0-9999)" 
                           min="0" max="9999" onchange="testInterface.saveAnswer(this.value)">
                </div>
                <small class="text-muted">Enter a whole number between 0 and 9999</small>
            </div>
        `;
    }
    
    renderMatchColumnInterface() {
        return `
            <div class="answer-options">
                <h6>Match Column I to Column II:</h6>
                <div class="row">
                    <div class="col-6">
                        <div class="mb-2">
                            <label class="form-label">A matches with:</label>
                            <select class="form-select" name="matchA" onchange="testInterface.saveMatchAnswer()">
                                <option value="">Select</option>
                                <option value="p">p</option>
                                <option value="q">q</option>
                                <option value="r">r</option>
                                <option value="s">s</option>
                            </select>
                        </div>
                        <div class="mb-2">
                            <label class="form-label">B matches with:</label>
                            <select class="form-select" name="matchB" onchange="testInterface.saveMatchAnswer()">
                                <option value="">Select</option>
                                <option value="p">p</option>
                                <option value="q">q</option>
                                <option value="r">r</option>
                                <option value="s">s</option>
                            </select>
                        </div>
                    </div>
                    <div class="col-6">
                        <div class="mb-2">
                            <label class="form-label">C matches with:</label>
                            <select class="form-select" name="matchC" onchange="testInterface.saveMatchAnswer()">
                                <option value="">Select</option>
                                <option value="p">p</option>
                                <option value="q">q</option>
                                <option value="r">r</option>
                                <option value="s">s</option>
                            </select>
                        </div>
                        <div class="mb-2">
                            <label class="form-label">D matches with:</label>
                            <select class="form-select" name="matchD" onchange="testInterface.saveMatchAnswer()">
                                <option value="">Select</option>
                                <option value="p">p</option>
                                <option value="q">q</option>
                                <option value="r">r</option>
                                <option value="s">s</option>
                            </select>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    getQuestionTypeBadgeColor(type) {
        switch(type) {
            case 'SCQ': return 'success';
            case 'MCQ': return 'info';
            case 'Integer': return 'warning';
            case 'MatchColumn': return 'secondary';
            default: return 'secondary';
        }
    }
    
    saveAnswer(answer) {
        const question = this.questions[this.currentQuestion - 1];
        this.answers.set(this.currentQuestion, answer);
        
        // Send to server
        this.sendAnswerToServer(question.id, answer);
        
        // Update navigation
        this.updateQuestionNavigation();
        this.updateProgress();
    }
    
    saveMCQAnswer() {
        const checkboxes = document.querySelectorAll('input[name="answer"]:checked');
        const answers = Array.from(checkboxes).map(cb => cb.value);
        
        const question = this.questions[this.currentQuestion - 1];
        this.answers.set(this.currentQuestion, answers);
        
        // Send to server
        this.sendAnswerToServer(question.id, answers);
        
        // Update navigation
        this.updateQuestionNavigation();
        this.updateProgress();
    }
    
    saveMatchAnswer() {
        const matches = {};
        ['A', 'B', 'C', 'D'].forEach(option => {
            const select = document.querySelector(`select[name="match${option}"]`);
            if (select && select.value) {
                matches[option] = select.value;
            }
        });
        
        const question = this.questions[this.currentQuestion - 1];
        this.answers.set(this.currentQuestion, matches);
        
        // Send to server
        this.sendAnswerToServer(question.id, matches);
        
        // Update navigation
        this.updateQuestionNavigation();
        this.updateProgress();
    }
    
    sendAnswerToServer(questionId, answer) {
        const timeSpent = this.timeSpent.get(this.currentQuestion) || 0;
        
        fetch('/api/save_answer', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: this.sessionId,
                question_id: questionId,
                answer: answer,
                time_spent: timeSpent
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.status !== 'success') {
                console.error('Error saving answer:', data.message);
            }
        })
        .catch(error => {
            console.error('Error saving answer:', error);
        });
    }
    
    loadSavedAnswer(question, answer) {
        switch(question.question_type) {
            case 'SCQ':
                const radio = document.querySelector(`input[name="answer"][value="${answer}"]`);
                if (radio) radio.checked = true;
                break;
                
            case 'MCQ':
                if (Array.isArray(answer)) {
                    answer.forEach(value => {
                        const checkbox = document.querySelector(`input[name="answer"][value="${value}"]`);
                        if (checkbox) checkbox.checked = true;
                    });
                }
                break;
                
            case 'Integer':
                const input = document.getElementById('integerAnswer');
                if (input) input.value = answer;
                break;
                
            case 'MatchColumn':
                if (typeof answer === 'object') {
                    Object.entries(answer).forEach(([key, value]) => {
                        const select = document.querySelector(`select[name="match${key}"]`);
                        if (select) select.value = value;
                    });
                }
                break;
        }
    }
    
    clearAnswer() {
        const question = this.questions[this.currentQuestion - 1];
        
        // Clear from UI
        switch(question.question_type) {
            case 'SCQ':
                document.querySelectorAll('input[name="answer"]').forEach(radio => radio.checked = false);
                break;
            case 'MCQ':
                document.querySelectorAll('input[name="answer"]').forEach(checkbox => checkbox.checked = false);
                break;
            case 'Integer':
                const input = document.getElementById('integerAnswer');
                if (input) input.value = '';
                break;
            case 'MatchColumn':
                document.querySelectorAll('select[name^="match"]').forEach(select => select.value = '');
                break;
        }
        
        // Clear from memory
        this.answers.delete(this.currentQuestion);
        
        // Update server
        this.sendAnswerToServer(question.id, null);
        
        // Update navigation
        this.updateQuestionNavigation();
        this.updateProgress();
    }
    
    flagQuestion(questionNumber) {
        const button = document.querySelector('.question-nav[data-question="' + questionNumber + '"]');
        if (button) {
            button.classList.toggle('flagged');
        }
    }
    
    showQuestion(questionNumber) {
        if (questionNumber >= 1 && questionNumber <= this.questions.length) {
            this.loadQuestion(questionNumber);
        }
    }
    
    nextQuestion() {
        if (this.currentQuestion < this.questions.length) {
            this.loadQuestion(this.currentQuestion + 1);
        }
    }
    
    previousQuestion() {
        if (this.currentQuestion > 1) {
            this.loadQuestion(this.currentQuestion - 1);
        }
    }
    
    updateNavigationButtons() {
        const prevBtn = document.getElementById('prevBtn');
        const nextBtn = document.getElementById('nextBtn');
        
        if (prevBtn) {
            prevBtn.disabled = this.currentQuestion === 1;
        }
        
        if (nextBtn) {
            nextBtn.disabled = this.currentQuestion === this.questions.length;
        }
    }
    
    updateQuestionNavigation() {
        const buttons = document.querySelectorAll('.question-nav');
        buttons.forEach((button, index) => {
            const questionNumber = index + 1;
            
            // Remove all state classes
            button.classList.remove('answered', 'current');
            
            // Add current class
            if (questionNumber === this.currentQuestion) {
                button.classList.add('current');
            }
            
            // Add answered class
            if (this.answers.has(questionNumber)) {
                button.classList.add('answered');
            }
        });
    }
    
    updateProgress() {
        const answeredCount = this.answers.size;
        const totalCount = this.questions.length;
        const percentage = (answeredCount / totalCount) * 100;
        
        const progressBar = document.getElementById('progressBar');
        const progressText = document.getElementById('progressText');
        
        if (progressBar) {
            progressBar.style.width = percentage + '%';
            progressBar.setAttribute('aria-valuenow', percentage);
        }
        
        if (progressText) {
            progressText.textContent = `${answeredCount} of ${totalCount} answered`;
        }
    }
    
    trackQuestionTime() {
        if (this.currentQuestion > 0) {
            const timeSpent = Date.now() - this.questionStartTime;
            const currentTime = this.timeSpent.get(this.currentQuestion) || 0;
            this.timeSpent.set(this.currentQuestion, currentTime + timeSpent);
        }
    }
    
    startAutoSave() {
        // Auto-save every 30 seconds
        setInterval(() => {
            this.answers.forEach((answer, questionNumber) => {
                const question = this.questions[questionNumber - 1];
                if (question) {
                    this.sendAnswerToServer(question.id, answer);
                }
            });
        }, 30000);
    }
    
    getTestSummary() {
        return {
            totalQuestions: this.questions.length,
            answered: this.answers.size,
            unanswered: this.questions.length - this.answers.size,
            timeElapsed: Date.now() - this.startTime,
            answers: Object.fromEntries(this.answers)
        };
    }
    
    cleanup() {
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
        }
        
        // Final time tracking
        this.trackQuestionTime();
        
        // Final auto-save
        this.answers.forEach((answer, questionNumber) => {
            const question = this.questions[questionNumber - 1];
            if (question) {
                this.sendAnswerToServer(question.id, answer);
            }
        });
    }
}

// Global functions for template usage
window.showQuestion = function(questionNumber) {
    if (window.testInterface) {
        window.testInterface.showQuestion(questionNumber);
    }
};

window.nextQuestion = function() {
    if (window.testInterface) {
        window.testInterface.nextQuestion();
    }
};

window.previousQuestion = function() {
    if (window.testInterface) {
        window.testInterface.previousQuestion();
    }
};

// Handle page unload
window.addEventListener('beforeunload', function() {
    if (window.testInterface) {
        window.testInterface.cleanup();
    }
});
