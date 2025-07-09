/**
 * Box Detection and Validation Interface
 * Handles interactive box detection, validation, and editing
 */

class BoxDetector {
    constructor(options) {
        this.containerId = options.containerId;
        this.pdfUrl = options.pdfUrl;
        this.detectedBoxes = options.detectedBoxes || [];
        this.onBoxChange = options.onBoxChange || function() {};
        this.onValidation = options.onValidation || function() {};
        
        this.viewer = null;
        this.editMode = false;
        this.selectedBox = null;
        this.isDrawing = false;
        this.startPoint = null;
        this.currentDrawBox = null;
        
        this.init();
    }
    
    async init() {
        await this.createInterface();
        this.setupEventListeners();
    }
    
    async createInterface() {
        const container = document.getElementById(this.containerId);
        if (!container) {
            console.error('Box detector container not found');
            return;
        }
        
        container.innerHTML = `
            <div class="box-detector-interface">
                <div class="controls-panel mb-3">
                    <div class="row">
                        <div class="col-md-6">
                            <div class="btn-group" role="group">
                                <button type="button" class="btn btn-outline-primary" id="viewMode">
                                    <i class="fas fa-eye"></i> View Mode
                                </button>
                                <button type="button" class="btn btn-outline-secondary" id="editMode">
                                    <i class="fas fa-edit"></i> Edit Mode
                                </button>
                                <button type="button" class="btn btn-outline-success" id="addBox">
                                    <i class="fas fa-plus"></i> Add Box
                                </button>
                            </div>
                        </div>
                        <div class="col-md-6 text-md-end">
                            <div class="btn-group" role="group">
                                <button type="button" class="btn btn-outline-info" id="validateBoxes">
                                    <i class="fas fa-check"></i> Validate All
                                </button>
                                <button type="button" class="btn btn-outline-warning" id="resetBoxes">
                                    <i class="fas fa-undo"></i> Reset
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="detection-stats mb-3">
                    <div class="row">
                        <div class="col-md-3">
                            <div class="stat-card">
                                <div class="stat-number" id="totalBoxes">${this.detectedBoxes.length}</div>
                                <div class="stat-label">Total Boxes</div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="stat-card">
                                <div class="stat-number" id="validBoxes">0</div>
                                <div class="stat-label">Valid Boxes</div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="stat-card">
                                <div class="stat-number" id="invalidBoxes">0</div>
                                <div class="stat-label">Invalid Boxes</div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="stat-card">
                                <div class="stat-number" id="averageConfidence">0%</div>
                                <div class="stat-label">Avg Confidence</div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="pdf-viewer-container">
                    <div id="pdfViewer" class="pdf-viewer"></div>
                </div>
                
                <div class="box-list mt-3">
                    <div class="card">
                        <div class="card-header">
                            <h5 class="mb-0">Detected Boxes</h5>
                        </div>
                        <div class="card-body">
                            <div id="boxList" class="box-list-container"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Initialize PDF viewer
        this.viewer = new PDFViewer('pdfViewer', this.pdfUrl);
        
        // Wait for viewer to load then set boxes
        setTimeout(() => {
            this.viewer.setBoxes(this.detectedBoxes);
            this.updateBoxList();
            this.updateStats();
        }, 1000);
    }
    
    setupEventListeners() {
        // Mode buttons
        document.getElementById('viewMode').addEventListener('click', () => this.setViewMode());
        document.getElementById('editMode').addEventListener('click', () => this.setEditMode());
        document.getElementById('addBox').addEventListener('click', () => this.startAddingBox());
        
        // Action buttons
        document.getElementById('validateBoxes').addEventListener('click', () => this.validateAllBoxes());
        document.getElementById('resetBoxes').addEventListener('click', () => this.resetBoxes());
        
        // PDF viewer mouse events for box drawing
        this.setupDrawingEvents();
    }
    
    setupDrawingEvents() {
        const pdfCanvas = document.getElementById('pdfViewer');
        if (!pdfCanvas) return;
        
        pdfCanvas.addEventListener('mousedown', (e) => this.handleMouseDown(e));
        pdfCanvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        pdfCanvas.addEventListener('mouseup', (e) => this.handleMouseUp(e));
        pdfCanvas.addEventListener('click', (e) => this.handleClick(e));
    }
    
    handleMouseDown(e) {
        if (!this.editMode) return;
        
        const rect = e.target.closest('.pdf-canvas-container').getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        // Check if clicking on existing box
        const clickedBox = this.getBoxAtPoint(x, y);
        if (clickedBox) {
            this.selectBox(clickedBox);
            return;
        }
        
        // Start drawing new box
        if (this.isDrawing) {
            this.startPoint = { x, y };
            this.createDrawBox(x, y);
        }
    }
    
    handleMouseMove(e) {
        if (!this.editMode || !this.isDrawing || !this.currentDrawBox) return;
        
        const rect = e.target.closest('.pdf-canvas-container').getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        this.updateDrawBox(x, y);
    }
    
    handleMouseUp(e) {
        if (!this.editMode || !this.isDrawing || !this.currentDrawBox) return;
        
        const rect = e.target.closest('.pdf-canvas-container').getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        this.finishDrawBox(x, y);
    }
    
    handleClick(e) {
        if (!this.editMode) return;
        
        const rect = e.target.closest('.pdf-canvas-container').getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        const clickedBox = this.getBoxAtPoint(x, y);
        if (clickedBox) {
            this.selectBox(clickedBox);
        }
    }
    
    getBoxAtPoint(x, y) {
        const currentPage = this.viewer.getCurrentPage();
        const scale = this.viewer.getScale();
        
        return this.detectedBoxes.find(box => {
            if (box.page !== currentPage) return false;
            
            const [x0, y0, x1, y1] = box.coordinates.map(coord => coord * scale);
            return x >= x0 && x <= x1 && y >= y0 && y <= y1;
        });
    }
    
    createDrawBox(x, y) {
        const overlay = document.querySelector('.pdf-overlay');
        if (!overlay) return;
        
        this.currentDrawBox = document.createElement('div');
        this.currentDrawBox.className = 'draw-box';
        this.currentDrawBox.style.position = 'absolute';
        this.currentDrawBox.style.border = '2px dashed #007bff';
        this.currentDrawBox.style.backgroundColor = 'rgba(0, 123, 255, 0.1)';
        this.currentDrawBox.style.left = x + 'px';
        this.currentDrawBox.style.top = y + 'px';
        this.currentDrawBox.style.width = '0px';
        this.currentDrawBox.style.height = '0px';
        
        overlay.appendChild(this.currentDrawBox);
    }
    
    updateDrawBox(x, y) {
        if (!this.currentDrawBox || !this.startPoint) return;
        
        const width = Math.abs(x - this.startPoint.x);
        const height = Math.abs(y - this.startPoint.y);
        const left = Math.min(x, this.startPoint.x);
        const top = Math.min(y, this.startPoint.y);
        
        this.currentDrawBox.style.left = left + 'px';
        this.currentDrawBox.style.top = top + 'px';
        this.currentDrawBox.style.width = width + 'px';
        this.currentDrawBox.style.height = height + 'px';
    }
    
    finishDrawBox(x, y) {
        if (!this.currentDrawBox || !this.startPoint) return;
        
        const width = Math.abs(x - this.startPoint.x);
        const height = Math.abs(y - this.startPoint.y);
        
        // Minimum size check
        if (width < 50 || height < 30) {
            this.currentDrawBox.remove();
            this.currentDrawBox = null;
            this.startPoint = null;
            return;
        }
        
        // Create new box
        const scale = this.viewer.getScale();
        const left = Math.min(x, this.startPoint.x) / scale;
        const top = Math.min(y, this.startPoint.y) / scale;
        const right = (Math.min(x, this.startPoint.x) + width) / scale;
        const bottom = (Math.min(y, this.startPoint.y) + height) / scale;
        
        const newBox = {
            page: this.viewer.getCurrentPage(),
            coordinates: [left, top, right, bottom],
            type: 'manual',
            question_number: this.detectedBoxes.length + 1,
            area: width * height,
            confidence: 1.0
        };
        
        this.detectedBoxes.push(newBox);
        
        // Clean up
        this.currentDrawBox.remove();
        this.currentDrawBox = null;
        this.startPoint = null;
        this.isDrawing = false;
        
        // Update interface
        this.viewer.setBoxes(this.detectedBoxes);
        this.updateBoxList();
        this.updateStats();
        this.onBoxChange(this.detectedBoxes);
    }
    
    selectBox(box) {
        this.selectedBox = box;
        this.highlightSelectedBox();
        this.showBoxDetails(box);
    }
    
    highlightSelectedBox() {
        // Remove previous highlights
        document.querySelectorAll('.box-overlay').forEach(el => {
            el.classList.remove('selected');
        });
        
        // Highlight selected box
        if (this.selectedBox) {
            this.viewer.highlightBox(this.selectedBox.question_number);
        }
    }
    
    showBoxDetails(box) {
        const modal = document.createElement('div');
        modal.className = 'modal fade';
        modal.innerHTML = `
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">Box Details</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <div class="row">
                            <div class="col-md-6">
                                <p><strong>Question Number:</strong> ${box.question_number}</p>
                                <p><strong>Page:</strong> ${box.page}</p>
                                <p><strong>Type:</strong> ${box.type}</p>
                                <p><strong>Area:</strong> ${Math.round(box.area)} px²</p>
                            </div>
                            <div class="col-md-6">
                                <p><strong>Coordinates:</strong></p>
                                <ul>
                                    <li>X: ${box.coordinates[0].toFixed(2)} - ${box.coordinates[2].toFixed(2)}</li>
                                    <li>Y: ${box.coordinates[1].toFixed(2)} - ${box.coordinates[3].toFixed(2)}</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-danger" onclick="boxDetector.deleteBox(${box.question_number})">Delete Box</button>
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        const bsModal = new bootstrap.Modal(modal);
        bsModal.show();
        
        // Clean up modal after hiding
        modal.addEventListener('hidden.bs.modal', () => {
            modal.remove();
        });
    }
    
    deleteBox(questionNumber) {
        this.detectedBoxes = this.detectedBoxes.filter(box => box.question_number !== questionNumber);
        
        // Renumber boxes
        this.detectedBoxes.forEach((box, index) => {
            box.question_number = index + 1;
        });
        
        this.viewer.setBoxes(this.detectedBoxes);
        this.updateBoxList();
        this.updateStats();
        this.onBoxChange(this.detectedBoxes);
        
        // Close modal
        const modal = document.querySelector('.modal.show');
        if (modal) {
            bootstrap.Modal.getInstance(modal).hide();
        }
    }
    
    setViewMode() {
        this.editMode = false;
        this.isDrawing = false;
        this.updateModeButtons();
    }
    
    setEditMode() {
        this.editMode = true;
        this.updateModeButtons();
    }
    
    startAddingBox() {
        this.setEditMode();
        this.isDrawing = true;
        
        // Visual feedback
        const pdfContainer = document.querySelector('.pdf-canvas-container');
        if (pdfContainer) {
            pdfContainer.style.cursor = 'crosshair';
        }
    }
    
    updateModeButtons() {
        const viewBtn = document.getElementById('viewMode');
        const editBtn = document.getElementById('editMode');
        
        if (this.editMode) {
            viewBtn.classList.remove('btn-outline-primary');
            viewBtn.classList.add('btn-outline-secondary');
            editBtn.classList.remove('btn-outline-secondary');
            editBtn.classList.add('btn-outline-primary');
        } else {
            viewBtn.classList.remove('btn-outline-secondary');
            viewBtn.classList.add('btn-outline-primary');
            editBtn.classList.remove('btn-outline-primary');
            editBtn.classList.add('btn-outline-secondary');
        }
    }
    
    validateAllBoxes() {
        const validBoxes = this.detectedBoxes.filter(box => this.isValidBox(box));
        const invalidBoxes = this.detectedBoxes.filter(box => !this.isValidBox(box));
        
        this.onValidation({
            total: this.detectedBoxes.length,
            valid: validBoxes.length,
            invalid: invalidBoxes.length,
            validBoxes: validBoxes,
            invalidBoxes: invalidBoxes
        });
        
        this.updateStats();
    }
    
    isValidBox(box) {
        const [x0, y0, x1, y1] = box.coordinates;
        const width = x1 - x0;
        const height = y1 - y0;
        
        // Basic validation rules
        return width > 50 && height > 30 && width < 2000 && height < 2000;
    }
    
    resetBoxes() {
        if (confirm('Are you sure you want to reset all boxes? This action cannot be undone.')) {
            this.detectedBoxes = [];
            this.viewer.setBoxes([]);
            this.updateBoxList();
            this.updateStats();
            this.onBoxChange([]);
        }
    }
    
    updateBoxList() {
        const listContainer = document.getElementById('boxList');
        if (!listContainer) return;
        
        if (this.detectedBoxes.length === 0) {
            listContainer.innerHTML = '<p class="text-muted">No boxes detected</p>';
            return;
        }
        
        const listHTML = this.detectedBoxes.map(box => `
            <div class="box-item d-flex justify-content-between align-items-center p-2 border rounded mb-2">
                <div>
                    <strong>Question ${box.question_number}</strong>
                    <small class="text-muted">Page ${box.page} | ${box.type}</small>
                </div>
                <div>
                    <span class="badge bg-${this.isValidBox(box) ? 'success' : 'danger'}">
                        ${this.isValidBox(box) ? 'Valid' : 'Invalid'}
                    </span>
                    <button class="btn btn-sm btn-outline-primary ms-2" onclick="boxDetector.goToBox(${box.question_number})">
                        <i class="fas fa-eye"></i>
                    </button>
                </div>
            </div>
        `).join('');
        
        listContainer.innerHTML = listHTML;
    }
    
    updateStats() {
        const total = this.detectedBoxes.length;
        const valid = this.detectedBoxes.filter(box => this.isValidBox(box)).length;
        const invalid = total - valid;
        const avgConfidence = total > 0 ? 
            Math.round(this.detectedBoxes.reduce((sum, box) => sum + (box.confidence || 0), 0) / total * 100) : 0;
        
        document.getElementById('totalBoxes').textContent = total;
        document.getElementById('validBoxes').textContent = valid;
        document.getElementById('invalidBoxes').textContent = invalid;
        document.getElementById('averageConfidence').textContent = avgConfidence + '%';
    }
    
    goToBox(questionNumber) {
        const box = this.detectedBoxes.find(b => b.question_number === questionNumber);
        if (box) {
            this.viewer.goToPage(box.page);
            setTimeout(() => {
                this.viewer.highlightBox(questionNumber);
            }, 300);
        }
    }
    
    // Public API methods
    getBoxes() {
        return this.detectedBoxes;
    }
    
    setBoxes(boxes) {
        this.detectedBoxes = boxes;
        this.viewer.setBoxes(boxes);
        this.updateBoxList();
        this.updateStats();
    }
    
    exportBoxes() {
        return {
            boxes: this.detectedBoxes,
            metadata: {
                total: this.detectedBoxes.length,
                valid: this.detectedBoxes.filter(box => this.isValidBox(box)).length,
                exportDate: new Date().toISOString()
            }
        };
    }
}

// Global functions
window.createBoxDetector = function(options) {
    return new BoxDetector(options);
};

// Global reference for template usage
window.boxDetector = null;
