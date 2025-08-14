// Robust Image Upload and Analysis System
class ImageUploadManager {
    constructor() {
        this.currentFile = null;
        this.analysisResult = null;
        this.isAnalyzing = false;
        this.init();
    }

    init() {
        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.setupElements());
        } else {
            this.setupElements();
        }
    }

    setupElements() {
        // Get all required elements
        this.uploadArea = document.getElementById('uploadArea');
        this.imageInput = document.getElementById('imageInput');
        this.uploadContent = document.querySelector('.upload-content');
        this.uploadPreview = document.getElementById('uploadPreview');
        this.previewImage = document.getElementById('previewImage');
        this.analyzeBtn = document.getElementById('analyzeBtn');
        this.analysisResults = document.getElementById('analysisResults');


        // Only proceed if all elements exist
        if (!this.uploadArea || !this.imageInput) {
            console.log('Image upload elements not found on this page');
            return;
        }

        this.setupEventListeners();
    }

    setupEventListeners() {
        // File input change event
        this.imageInput.addEventListener('change', (e) => {
            this.handleFileSelection(e.target.files);
        });

        // Upload area click event (excluding the button)
        this.uploadArea.addEventListener('click', (e) => {
            // Don't trigger if clicking the button or its children
            if (e.target.closest('button') || e.target.tagName === 'BUTTON') {
                return; // Let the button handle its own click
            }
            
            if (!this.isAnalyzing && this.uploadContent && !this.uploadContent.classList.contains('d-none')) {
                e.preventDefault();
                this.imageInput.click();
            }
        });

        // Drag and drop events
        this.setupDragAndDrop();

        // Analyze button event
        if (this.analyzeBtn) {
            this.analyzeBtn.addEventListener('click', () => {
                this.analyzeImage();
            });
        }

        // Reset button events
        const resetBtns = document.querySelectorAll('[onclick="resetImageAnalysis()"], [onclick="clearImagePreview()"]');
        resetBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                this.resetUpload();
            });
        });
    }

    setupDragAndDrop() {
        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            this.uploadArea.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
            });
        });

        // Highlight drop area when item is dragged over it
        ['dragenter', 'dragover'].forEach(eventName => {
            this.uploadArea.addEventListener(eventName, () => {
                if (!this.isAnalyzing) {
                    this.uploadArea.classList.add('dragover');
                }
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            this.uploadArea.addEventListener(eventName, () => {
                this.uploadArea.classList.remove('dragover');
            });
        });

        // Handle dropped files
        this.uploadArea.addEventListener('drop', (e) => {
            if (!this.isAnalyzing) {
                const dt = e.dataTransfer;
                const files = dt.files;
                this.handleFileSelection(files);
            }
        });
    }

    handleFileSelection(files) {
        if (!files || files.length === 0) {
            return;
        }

        const file = files[0];
        
        // Validate file type
        if (!this.isValidImageFile(file)) {
            this.showError('Please select a valid image file (JPG, JPEG, PNG, or GIF)');
            return;
        }

        // Validate file size (20MB limit)
        if (file.size > 20 * 1024 * 1024) {
            this.showError('Image file is too large. Please select an image under 20MB.');
            return;
        }

        this.currentFile = file;
        this.displayImagePreview(file);
    }

    isValidImageFile(file) {
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif'];
        return validTypes.includes(file.type);
    }

    displayImagePreview(file) {
        const reader = new FileReader();
        
        reader.onload = (e) => {
            if (this.previewImage) {
                this.previewImage.src = e.target.result;
                this.previewImage.style.maxHeight = '200px';
                this.previewImage.style.maxWidth = '100%';
            }
            
            // Show preview, hide upload content
            if (this.uploadContent) {
                this.uploadContent.classList.add('d-none');
            }
            if (this.uploadPreview) {
                this.uploadPreview.classList.remove('d-none');
            }
            
            // Enable analyze button
            if (this.analyzeBtn) {
                this.analyzeBtn.disabled = false;
            }
            
            // Hide previous analysis results
            if (this.analysisResults) {
                this.analysisResults.classList.add('d-none');
            }
        };
        
        reader.onerror = () => {
            this.showError('Failed to read the image file. Please try again.');
        };
        
        reader.readAsDataURL(file);
    }

    async analyzeImage() {
        if (!this.currentFile || this.isAnalyzing) {
            return;
        }

        this.isAnalyzing = true;
        this.setLoadingState(true);

        try {
            // Create FormData
            const formData = new FormData();
            formData.append('image', this.currentFile);

            // Send to analysis API
            const response = await fetch('/api/analyze-image', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                this.analysisResult = result;
                this.displayAnalysisResults(result.analysis);
                this.showSuccess('Image analyzed successfully!');
            } else {
                this.showError(result.error || 'Analysis failed. Please try again.');
            }

        } catch (error) {
            console.error('Analysis error:', error);
            this.showError('Failed to analyze image. Please check your connection and try again.');
        } finally {
            this.isAnalyzing = false;
            this.setLoadingState(false);
        }
    }

    setLoadingState(isLoading) {
        if (!this.analyzeBtn) return;

        const analyzeText = this.analyzeBtn.querySelector('.analyze-btn-text');
        const analyzeLoading = this.analyzeBtn.querySelector('.analyze-btn-loading');

        if (isLoading) {
            if (analyzeText) analyzeText.classList.add('d-none');
            if (analyzeLoading) analyzeLoading.classList.remove('d-none');
            this.analyzeBtn.disabled = true;
        } else {
            if (analyzeText) analyzeText.classList.remove('d-none');
            if (analyzeLoading) analyzeLoading.classList.add('d-none');
            this.analyzeBtn.disabled = false;
        }
    }

    displayAnalysisResults(analysis) {
        if (!this.analysisResults) return;

        // Update analysis result elements
        const elements = {
            'detectedProduct': analysis.product_name || 'Unknown',
            'detectedBrand': analysis.brand || 'Not detected',
            'detectedCategory': analysis.category || 'Unknown',
            'detectedFeatures': (analysis.features || []).join(', ') || 'None detected'
        };

        Object.entries(elements).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = value;
            }
        });

        // Update confidence display
        const confidence = Math.round((analysis.confidence || 0) * 100);
        const confidenceText = document.getElementById('confidenceText');
        const confidenceBar = document.getElementById('confidenceBar');
        
        if (confidenceText) confidenceText.textContent = confidence + '%';
        if (confidenceBar) {
            confidenceBar.style.width = confidence + '%';
            confidenceBar.className = `progress-bar ${this.getConfidenceClass(confidence)}`;
        }

        // Show results
        this.analysisResults.classList.remove('d-none');
    }

    getConfidenceClass(confidence) {
        if (confidence >= 80) return 'bg-success';
        if (confidence >= 60) return 'bg-warning';
        return 'bg-danger';
    }

    resetUpload() {
        this.currentFile = null;
        this.analysisResult = null;
        this.isAnalyzing = false;

        // Reset file input
        if (this.imageInput) {
            this.imageInput.value = '';
        }

        // Show upload content, hide preview
        if (this.uploadContent) {
            this.uploadContent.classList.remove('d-none');
        }
        if (this.uploadPreview) {
            this.uploadPreview.classList.add('d-none');
        }

        // Hide analysis results
        if (this.analysisResults) {
            this.analysisResults.classList.add('d-none');
        }

        // Disable analyze button
        if (this.analyzeBtn) {
            this.analyzeBtn.disabled = true;
        }

        // Remove any error/success messages
        this.clearMessages();
    }

    searchFromAnalysis() {
        if (!this.analysisResult) {
            this.showError('No analysis data available');
            return;
        }

        const analysis = this.analysisResult.analysis;
        const searchQuery = this.analysisResult.search_query;
        
        // Build URL with analysis data
        const params = new URLSearchParams({
            query: searchQuery,
            image_search: 'true',
            detected_product: analysis.product_name || '',
            detected_brand: analysis.brand || '',
            detected_category: analysis.category || '',
            detected_features: (analysis.features || []).join(','),
            confidence: (analysis.confidence || 0).toString()
        });
        
        // Navigate to search with analysis data
        window.location.href = '/search?' + params.toString();
    }

    showError(message) {
        this.showMessage(message, 'error');
    }

    showSuccess(message) {
        this.showMessage(message, 'success');
    }

    showMessage(message, type) {
        // Remove existing messages
        this.clearMessages();

        // Create alert element
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type === 'error' ? 'danger' : 'success'} alert-dismissible fade show mt-3`;
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        // Insert after upload area
        if (this.uploadArea && this.uploadArea.parentNode) {
            this.uploadArea.parentNode.insertBefore(alertDiv, this.uploadArea.nextSibling);
        }

        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }

    clearMessages() {
        // Remove any existing alert messages
        const alerts = document.querySelectorAll('.alert');
        alerts.forEach(alert => {
            if (alert.parentNode) {
                alert.remove();
            }
        });
    }
}

// Global functions for backward compatibility
function resetImageAnalysis() {
    if (window.imageUploadManager) {
        window.imageUploadManager.resetUpload();
    }
}

function clearImagePreview() {
    if (window.imageUploadManager) {
        window.imageUploadManager.resetUpload();
    }
}

function searchFromAnalysis() {
    if (window.imageUploadManager) {
        window.imageUploadManager.searchFromAnalysis();
    }
}

function analyzeImage() {
    if (window.imageUploadManager) {
        window.imageUploadManager.analyzeImage();
    }
}

// Initialize the image upload manager
window.imageUploadManager = new ImageUploadManager();