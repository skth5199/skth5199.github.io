/**
 * Shared navigation component.
 * Usage: add <div id="main-nav"></div> and <script src="./nav.js"></script> to any page.
 * The script auto-detects the current page and highlights the active nav link.
 */
(function () {
    const pages = [
        { label: 'About',    href: './about.html' },
        { label: 'Projects', href: './projects.html' },
        { label: 'Blog',     href: './index.html#blog' },
        { label: 'AI Agent', href: './agent.html' },
        { label: 'Contact',  href: './index.html#contact' },
    ];

    const currentPath = window.location.pathname.split('/').pop() || 'index.html';

    function isActive(href) {
        const hrefPage = href.split('/').pop().split('#')[0];
        if (hrefPage === currentPath) return true;
        // "Blog" and "Contact" live on index.html — only highlight if we're on index
        if (currentPath === 'index.html' && hrefPage === 'index.html') return false;
        return false;
    }

    const linksHTML = pages.map(p => {
        const style = isActive(p.href) ? ' style="color: var(--text-primary);"' : '';
        return `<li><a href="${p.href}"${style}>${p.label}</a></li>`;
    }).join('\n                ');

    const nav = document.getElementById('main-nav');
    if (!nav) return;

    nav.outerHTML = `
    <nav class="nav">
        <div class="container">
            <a href="./index.html" class="nav-logo">Sri.</a>
            <ul class="nav-links" id="navLinks">
                ${linksHTML}
                <li><a href="https://topmate.io/s_srikanth" target="_blank" class="btn btn-primary">Book a Call</a></li>
            </ul>
            <div class="hamburger" id="hamburger" onclick="toggleNav()">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    </nav>`;

    // Mobile menu toggle
    window.toggleNav = function () {
        document.getElementById('navLinks').classList.toggle('open');
    };
    document.querySelectorAll('.nav-links a').forEach(function (link) {
        link.addEventListener('click', function () {
            document.getElementById('navLinks').classList.remove('open');
        });
    });
})();
