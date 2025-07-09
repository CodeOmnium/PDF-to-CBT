/**
 * PDF Viewer with Box Detection Visualization
 * Handles PDF display and box overlay rendering
 */

class PDFViewer {
    constructor(containerId, pdfUrl) {
        this.container = document.getElementById(containerId);
        this.pdfUrl = pdfUrl;
        this.pdf = null;
        this.currentPage = 1;
        this.scale = 1.0;
        this.boxes = [];
        this.canvasElements = [];
        
        this.init();
    }
    
    async init() {
        if (!this.container) {
            console.error('PDF viewer container not found');
            return;
        }
        
        try {
            // Initialize PDF.js
            pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
            
            // Load PDF
            this.pdf = await pdfjsLib.getDocument(this.pdfUrl).promise;
            
            // Create viewer interface
            this.createViewer();
            
            // Load first page
            await this.loadPage(1);
            
        } catch (error) {
            console.error('Error initializing PDF viewer:', error);
            this.showError('Failed to load PDF');
        }
    }
    
    createViewer() {
        this.container.innerHTML = `
            <div class="pdf-viewer-controls mb-3">
                <div class="btn-group" role="group">
                    <button type="button" class="btn btn-outline-secondary" id="prevPage">
                        <i class="fas fa-chevron-left"></i> Previous
                    </button>
                    <button type="button" class="btn btn-outline-secondary" id="nextPage">
                        Next <i class="fas fa-chevron-right"></i>
                    </button>
                </div>
                <div class="btn-group ms-2" role="group">
                    <button type="button" class="btn btn-outline-secondary" id="zoomOut">
                        <i class="fas fa-minus"></i>
                    </button>
                    <button type="button" class="btn btn-outline-secondary" id="zoomIn">
                        <i class="fas fa-plus"></i>
                    </button>
                </div>
                <span class="ms-3">
                    Page <span id="pageNum">1</span> of <span id="pageCount">${this.pdf.numPages}</span>
                    | Scale: <span id="scalePercent">100%</span>
                </span>
            </div>
            <div class="pdf-viewer-content">
                <div id="pdfCanvas" class="pdf-canvas-container"></div>
            </div>
        `;
        
        // Add event listeners
        this.setupEventListeners();
    }
    
    setupEventListeners() {
        document.getElementById('prevPage').addEventListener('click', () => this.previousPage());
        document.getElementById('nextPage').addEventListener('click', () => this.nextPage());
        document.getElementById('zoomIn').addEventListener('click', () => this.zoomIn());
        document.getElementById('zoomOut').addEventListener('click', () => this.zoomOut());
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.target.closest('.pdf-viewer') || e.target === document.body) {
                switch(e.key) {
                    case 'ArrowLeft':
                        e.preventDefault();
                        this.previousPage();
                        break;
                    case 'ArrowRight':
                        e.preventDefault();
                        this.nextPage();
                        break;
                    case '+':
                    case '=':
                        e.preventDefault();
                        this.zoomIn();
                        break;
                    case '-':
                        e.preventDefault();
                        this.zoomOut();
                        break;
                }
            }
        });
    }
    
    async loadPage(pageNumber) {
        if (pageNumber < 1 || pageNumber > this.pdf.numPages) {
            return;
        }
        
        try {
            this.currentPage = pageNumber;
            
            // Get page
            const page = await this.pdf.getPage(pageNumber);
            
            // Set up canvas
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            // Calculate viewport
            const viewport = page.getViewport({ scale: this.scale });
            canvas.width = viewport.width;
            canvas.height = viewport.height;
            
            // Render page
            const renderContext = {
                canvasContext: ctx,
                viewport: viewport
            };
            
            await page.render(renderContext).promise;
            
            // Update container
            const canvasContainer = document.getElementById('pdfCanvas');
            canvasContainer.innerHTML = '';
            canvasContainer.appendChild(canvas);
            
            // Store canvas reference
            this.canvasElements[pageNumber - 1] = canvas;
            
            // Update UI
            this.updateUI();
            
            // Render boxes for this page
            this.renderBoxes(pageNumber);
            
        } catch (error) {
            console.error('Error loading page:', error);
            this.showError('Failed to load page');
        }
    }
    
    renderBoxes(pageNumber) {
        const pageBoxes = this.boxes.filter(box => box.page === pageNumber);
        if (pageBoxes.length === 0) return;
        
        const canvasContainer = document.getElementById('pdfCanvas');
        const canvas = canvasContainer.querySelector('canvas');
        if (!canvas) return;
        
        // Create overlay div
        const overlay = document.createElement('div');
        overlay.className = 'pdf-overlay';
        overlay.style.position = 'absolute';
        overlay.style.top = '0';
        overlay.style.left = '0';
        overlay.style.width = canvas.width + 'px';
        overlay.style.height = canvas.height + 'px';
        overlay.style.pointerEvents = 'none';
        
        // Add boxes
        pageBoxes.forEach(box => {
            const boxElement = this.createBoxElement(box);
            overlay.appendChild(boxElement);
        });
        
        // Position canvas container relatively
        canvasContainer.style.position = 'relative';
        canvasContainer.appendChild(overlay);
    }
    
    createBoxElement(box) {
        const boxDiv = document.createElement('div');
        boxDiv.className = 'box-overlay';
        
        // Scale coordinates
        const scaledCoords = box.coordinates.map(coord => coord * this.scale);
        const [x0, y0, x1, y1] = scaledCoords;
        
        boxDiv.style.left = x0 + 'px';
        boxDiv.style.top = y0 + 'px';
        boxDiv.style.width = (x1 - x0) + 'px';
        boxDiv.style.height = (y1 - y0) + 'px';
        
        // Add label
        const label = document.createElement('div');
        label.className = 'box-label';
        label.textContent = box.question_number || 'Box';
        boxDiv.appendChild(label);
        
        return boxDiv;
    }
    
    setBoxes(boxes) {
        this.boxes = boxes;
        this.renderBoxes(this.currentPage);
    }
    
    previousPage() {
        if (this.currentPage > 1) {
            this.loadPage(this.currentPage - 1);
        }
    }
    
    nextPage() {
        if (this.currentPage < this.pdf.numPages) {
            this.loadPage(this.currentPage + 1);
        }
    }
    
    zoomIn() {
        this.scale = Math.min(this.scale * 1.2, 3.0);
        this.loadPage(this.currentPage);
    }
    
    zoomOut() {
        this.scale = Math.max(this.scale / 1.2, 0.5);
        this.loadPage(this.currentPage);
    }
    
    updateUI() {
        document.getElementById('pageNum').textContent = this.currentPage;
        document.getElementById('scalePercent').textContent = Math.round(this.scale * 100) + '%';
        
        // Update button states
        document.getElementById('prevPage').disabled = this.currentPage === 1;
        document.getElementById('nextPage').disabled = this.currentPage === this.pdf.numPages;
    }
    
    showError(message) {
        this.container.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle me-2"></i>
                ${message}
            </div>
        `;
    }
    
    // Public API methods
    goToPage(pageNumber) {
        this.loadPage(pageNumber);
    }
    
    setScale(scale) {
        this.scale = Math.max(0.5, Math.min(3.0, scale));
        this.loadPage(this.currentPage);
    }
    
    getCurrentPage() {
        return this.currentPage;
    }
    
    getTotalPages() {
        return this.pdf ? this.pdf.numPages : 0;
    }
    
    getScale() {
        return this.scale;
    }
    
    highlightBox(questionNumber) {
        const box = this.boxes.find(b => b.question_number === questionNumber);
        if (box) {
            // Go to the page containing this box
            if (box.page !== this.currentPage) {
                this.loadPage(box.page);
            }
            
            // Highlight the box
            setTimeout(() => {
                const boxElements = document.querySelectorAll('.box-overlay');
                boxElements.forEach(el => {
                    const label = el.querySelector('.box-label');
                    if (label && label.textContent === questionNumber.toString()) {
                        el.style.borderColor = '#ff6b6b';
                        el.style.backgroundColor = 'rgba(255, 107, 107, 0.2)';
                        
                        // Reset after 3 seconds
                        setTimeout(() => {
                            el.style.borderColor = '';
                            el.style.backgroundColor = '';
                        }, 3000);
                    }
                });
            }, 100);
        }
    }
}

// Utility functions
window.createPDFViewer = function(containerId, pdfUrl) {
    return new PDFViewer(containerId, pdfUrl);
};

window.highlightQuestionBox = function(viewer, questionNumber) {
    if (viewer && viewer.highlightBox) {
        viewer.highlightBox(questionNumber);
    }
};
