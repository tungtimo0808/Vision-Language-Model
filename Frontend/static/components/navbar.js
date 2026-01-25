class CustomNavbar extends HTMLElement {
  connectedCallback() {
    this.attachShadow({ mode: 'open' });
    this.shadowRoot.innerHTML = `
      <style>
        /* CSS Reset nhỏ cho Shadow DOM */
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: system-ui, -apple-system, sans-serif; }
        
        nav {
            background: white;
            border-bottom: 1px solid #f3f4f6;
            padding: 1rem 0;
        }

        .container {
            max-width: 1280px;
            margin: 0 auto;
            padding: 0 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        a { text-decoration: none; }

        .brand {
            font-size: 1.25rem;
            font-weight: 800;
            color: #be123c; /* Primary-700 (Red) */
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .menu-btn {
            display: none; /* Ẩn trên desktop */
            background: none;
            border: none;
            cursor: pointer;
            color: #374151;
        }

        /* Hover effect màu đỏ */
        .nav-link:hover {
          color: #e11d48;
        }
        
        .nav-link::after {
          content: '';
          display: block;
          width: 0;
          height: 2px;
          background: #e11d48; /* Primary-600 (Red) */
          transition: width 0.3s;
          margin-top: 2px;
        }
        
        .nav-link:hover::after {
          width: 100%;
        }

        @media (max-width: 768px) {
            .menu-btn { display: block; }
        }
      </style>

      <nav>
        <div class="container">
          <a href="/" class="brand">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: #e11d48;">
                <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
                <circle cx="12" cy="12" r="3"></circle>
            </svg>
            GalLens AI
          </a>

          <button class="menu-btn">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <line x1="3" y1="12" x2="21" y2="12"></line>
              <line x1="3" y1="6" x2="21" y2="6"></line>
              <line x1="3" y1="18" x2="21" y2="18"></line>
            </svg>
          </button>
        </div>
      </nav>
    `;
  }
}

customElements.define('custom-navbar', CustomNavbar);