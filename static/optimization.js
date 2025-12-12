// Frontend optimizations
document.addEventListener('DOMContentLoaded', function() {
    // Lazy load images
    const images = document.querySelectorAll('img[data-src]');
    const imageObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                img.src = img.dataset.src;
                img.removeAttribute('data-src');
                observer.unobserve(img);
            }
        });
    });
    
    images.forEach(img => imageObserver.observe(img));
    
    // Prefetch pages on hover
    const links = document.querySelectorAll('a[href^="/"]');
    links.forEach(link => {
        link.addEventListener('mouseenter', () => {
            const prefetchLink = document.createElement('link');
            prefetchLink.rel = 'prefetch';
            prefetchLink.href = link.href;
            document.head.appendChild(prefetchLink);
        });
    });
    
    // Cache API responses
    const cache = {};
    window.cachedFetch = async function(url, options = {}) {
        const cacheKey = url + JSON.stringify(options);
        if (cache[cacheKey] && Date.now() - cache[cacheKey].timestamp < 300000) {
            return cache[cacheKey].data;
        }
        
        const response = await fetch(url, options);
        const data = await response.json();
        cache[cacheKey] = {
            data: data,
            timestamp: Date.now()
        };
        return data;
    };
});
