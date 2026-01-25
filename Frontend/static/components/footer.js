class CustomFooter extends HTMLElement {
  connectedCallback() {
    this.attachShadow({ mode: 'open' });
    this.shadowRoot.innerHTML = `
      <style>
        /* CSS Reset cho Shadow DOM */
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: system-ui, -apple-system, sans-serif; }

        footer {
            background-color: #1f2937; /* Gray-800 */
            color: white;
            padding: 3rem 0;
            margin-top: auto;
        }

        .container {
            max-width: 1280px;
            margin: 0 auto;
            padding: 0 1rem;
        }

        .grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 2rem;
        }

        @media (min-width: 768px) {
            .grid { grid-template-columns: repeat(4, 1fr); }
        }

        h3, h4 { margin-bottom: 1rem; font-weight: bold; }
        h3 { font-size: 1.125rem; }

        p, li, a { color: #9ca3af; /* Gray-400 */ font-size: 0.95rem; line-height: 1.5; }
        
        ul { list-style: none; }
        
        .flex-items { display: flex; align-items: center; gap: 0.5rem; }
        .flex-gap { display: flex; gap: 1rem; margin-top: 1rem; }

        /* Hover effect màu đỏ */
        a { text-decoration: none; transition: color 0.2s; }
        a:hover, .footer-link:hover {
          color: #e11d48; /* Đổi từ Green sang Red */
        }

        .border-t {
            border-top: 1px solid #374151;
            margin-top: 2rem;
            padding-top: 2rem;
            text-align: center;
        }
      </style>

      <footer>
        <div class="container">
          <div class="grid">
            <div>
              <h3>GalLens AI</h3>
              <p>AI-powered chicken health diagnostics for poultry farmers and backyard chicken enthusiasts.</p>
            </div>

            <div style="grid-column: span 3;">
              <h4>Contact</h4>
              <ul>
                <li class="flex-items">
                  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"></path>
                    <polyline points="22,6 12,13 2,6"></polyline>
                  </svg>
                  <span>mustela410@gmail.com</span>
                </li>
              </ul>
              
              <div class="flex-gap">
                <a href="#">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M18 2h-3a5 5 0 0 0-5 5v3H7v4h3v8h4v-8h3l1-4h-4V7a1 1 0 0 1 1-1h3z"></path>
                  </svg>
                </a>
                <a href="#">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M23 3a10.9 10.9 0 0 1-3.14 1.53 4.48 4.48 0 0 0-7.86 3v1A10.66 10.66 0 0 1 3 4s-4 9 5 13a11.64 11.64 0 0 1-7 2c9 5 20 0 20-11.5a4.5 4.5 0 0 0-.08-.83A7.72 7.72 0 0 0 23 3z"></path>
                  </svg>
                </a>
                <a href="#">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="2" y="2" width="20" height="20" rx="5" ry="5"></rect>
                    <path d="M16 11.37A4 4 0 1 1 12.63 8 4 4 0 0 1 16 11.37z"></path>
                    <line x1="17.5" y1="6.5" x2="17.51" y2="6.5"></line>
                  </svg>
                </a>
              </div>
            </div>
          </div>

          <div class="border-t">
            <p>© 2025 Group Project VLM.</p>
          </div>
        </div>
      </footer>
    `;
  }
}

customElements.define('custom-footer', CustomFooter);