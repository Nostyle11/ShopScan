// Main script file for PriceHunter

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Handle search form submission
    const searchForm = document.querySelector('form.search-form');
    if (searchForm) {
        searchForm.addEventListener('submit', function(e) {
            const searchInput = this.querySelector('input[name="query"]');
            
            if (!searchInput.value.trim()) {
                e.preventDefault();
                showAlert('Please enter a search term', 'warning');
                return;
            }
        });
    }

    // Function to show alert
    function showAlert(message, type = 'info') {
        const alertContainer = document.createElement('div');
        alertContainer.className = `alert alert-${type} alert-dismissible fade show`;
        alertContainer.role = 'alert';
        
        alertContainer.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        const mainContent = document.querySelector('main.container');
        if (mainContent) {
            mainContent.prepend(alertContainer);
            
            // Auto-dismiss after 5 seconds
            setTimeout(() => {
                const bsAlert = new bootstrap.Alert(alertContainer);
                bsAlert.close();
            }, 5000);
        }
    }

    // Handle "Back to Top" button
    const backToTopBtn = document.getElementById('backToTopBtn');
    if (backToTopBtn) {
        window.addEventListener('scroll', function() {
            if (document.body.scrollTop > 20 || document.documentElement.scrollTop > 20) {
                backToTopBtn.style.display = 'block';
            } else {
                backToTopBtn.style.display = 'none';
            }
        });

        backToTopBtn.addEventListener('click', function() {
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        });
    }

    // Format currency
    const formatCurrency = (amount) => {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD'
        }).format(amount);
    };

    // Highlight price differences in comparison tables
    const priceTable = document.querySelector('.table');
    if (priceTable) {
        const priceRows = priceTable.querySelectorAll('tbody tr');
        let lowestPrice = Infinity;
        let lowestPriceIndex = -1;

        // Find the lowest price
        priceRows.forEach((row, index) => {
            const priceCell = row.querySelector('td:nth-child(2)');
            if (priceCell) {
                const priceText = priceCell.textContent.trim();
                const price = parseFloat(priceText.replace('$', '').replace(',', ''));
                
                if (price && price < lowestPrice) {
                    lowestPrice = price;
                    lowestPriceIndex = index;
                }
            }
        });

        // Highlight the row with the lowest price
        if (lowestPriceIndex >= 0) {
            priceRows[lowestPriceIndex].classList.add('table-success');
        }
    }

    // Theme Toggle Functionality
    function initThemeToggle() {
        const themeToggle = document.getElementById('themeToggle');
        const themeIcon = document.getElementById('themeIcon');
        const themeText = document.getElementById('themeText');
        const html = document.documentElement;
        
        // Get current theme from localStorage or default to dark
        let currentTheme = localStorage.getItem('theme') || 'dark';
        
        // Apply the current theme
        function applyTheme(theme) {
            html.setAttribute('data-bs-theme', theme);
            if (theme === 'dark') {
                themeIcon.className = 'fas fa-sun';
                themeText.textContent = 'Light';
            } else {
                themeIcon.className = 'fas fa-moon';
                themeText.textContent = 'Dark';
            }
            localStorage.setItem('theme', theme);
        }
        
        // Initialize theme
        applyTheme(currentTheme);
        
        // Toggle theme on button click
        if (themeToggle) {
            themeToggle.addEventListener('click', () => {
                currentTheme = currentTheme === 'dark' ? 'light' : 'dark';
                applyTheme(currentTheme);
            });
        }
    }

    // Update footer year
    function updateFooterYear() {
        const yearSpan = document.getElementById('currentYear');
        if (yearSpan) {
            yearSpan.textContent = new Date().getFullYear();
        }
    }

    // Initialize theme toggle and footer year
    initThemeToggle();
    updateFooterYear();

    // Theme Toggle Functionality
    function initThemeToggle() {
        const themeToggle = document.getElementById('themeToggle');
        const themeIcon = document.getElementById('themeIcon');
        const themeText = document.getElementById('themeText');
        const html = document.documentElement;
        
        // Get current theme from localStorage or default to dark
        let currentTheme = localStorage.getItem('theme') || 'dark';
        
        // Apply the current theme
        function applyTheme(theme) {
            html.setAttribute('data-bs-theme', theme);
            if (theme === 'dark') {
                if (themeIcon) themeIcon.className = 'fas fa-sun';
                if (themeText) themeText.textContent = 'Light';
            } else {
                if (themeIcon) themeIcon.className = 'fas fa-moon';
                if (themeText) themeText.textContent = 'Dark';
            }
            localStorage.setItem('theme', theme);
        }
        
        // Initialize theme
        applyTheme(currentTheme);
        
        // Toggle theme on button click
        if (themeToggle) {
            themeToggle.addEventListener('click', () => {
                currentTheme = currentTheme === 'dark' ? 'light' : 'dark';
                applyTheme(currentTheme);
            });
        }
    }

    // Update footer year
    function updateFooterYear() {
        const yearSpan = document.getElementById('currentYear');
        if (yearSpan) {
            yearSpan.textContent = new Date().getFullYear();
        }
    }

    // Product filtering functionality
    function initProductFilters() {
        // Filter products based on price and retailer
        function filterProducts() {
            const minPrice = parseFloat(document.getElementById('minPrice')?.value) || 0;
            const maxPrice = parseFloat(document.getElementById('maxPrice')?.value) || Infinity;
            const selectedRetailers = Array.from(document.querySelectorAll('.retailer-filter:checked'))
                .map(cb => cb.value);

            const productCards = document.querySelectorAll('.product-card');
            let visibleCount = 0;

            productCards.forEach(card => {
                const price = parseFloat(card.dataset.price) || 0;
                const retailer = card.dataset.retailer || '';
                
                const priceMatch = price >= minPrice && price <= maxPrice;
                const retailerMatch = selectedRetailers.length === 0 || selectedRetailers.includes(retailer);
                
                if (priceMatch && retailerMatch) {
                    card.parentElement.style.display = 'block';
                    visibleCount++;
                } else {
                    card.parentElement.style.display = 'none';
                }
            });

            // Update product count
            const productCount = document.getElementById('productCount');
            if (productCount) {
                productCount.textContent = visibleCount;
            }
        }

        // Add event listeners for price filters
        const minPriceInput = document.getElementById('minPrice');
        const maxPriceInput = document.getElementById('maxPrice');
        const retailerCheckboxes = document.querySelectorAll('.retailer-filter');
        
        if (minPriceInput) minPriceInput.addEventListener('input', filterProducts);
        if (maxPriceInput) maxPriceInput.addEventListener('input', filterProducts);
        
        retailerCheckboxes.forEach(checkbox => {
            checkbox.addEventListener('change', filterProducts);
        });

        // Product sorting functionality
        const sortButtons = document.querySelectorAll('.sort-btn');
        sortButtons.forEach(button => {
            button.addEventListener('click', () => {
                const sortType = button.dataset.sort;
                const productContainer = document.querySelector('.row');
                const products = Array.from(productContainer.querySelectorAll('.col-6'));
                
                products.sort((a, b) => {
                    const priceA = parseFloat(a.querySelector('.product-card').dataset.price) || 0;
                    const priceB = parseFloat(b.querySelector('.product-card').dataset.price) || 0;
                    
                    if (sortType === 'price-asc') {
                        return priceA - priceB;
                    } else if (sortType === 'price-desc') {
                        return priceB - priceA;
                    }
                    return 0;
                });
                
                // Remove and re-append sorted products
                products.forEach(product => product.remove());
                products.forEach(product => productContainer.appendChild(product));
                
                // Update button states
                sortButtons.forEach(btn => btn.classList.remove('active'));
                button.classList.add('active');
            });
        });
    }

    // Initialize all functionality
    initThemeToggle();
    updateFooterYear();
    initProductFilters();
});
