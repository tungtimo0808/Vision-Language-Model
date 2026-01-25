document.addEventListener('DOMContentLoaded', function() {
    const imageInput = document.getElementById('imageInput');
    const chatInput = document.getElementById('chatInput');
    const sendBtn = document.getElementById('sendBtn');
    const uploadBtn = document.getElementById('uploadBtn');
    const chatContainer = document.getElementById('chatContainer');
    const newChatBtn = document.getElementById('newChatBtn');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const clearImageBtn = document.getElementById('clearImageBtn');
    const welcomeScreen = document.getElementById('welcomeScreen');
    const modelSelectChat = document.getElementById('modelSelectChat');
    const currentModelBadge = document.getElementById('currentModelBadge');
    
    const BOT_AVATAR = "https://i.pinimg.com/736x/d4/81/a3/d481a371c6443f0e5f8e84c7a5d9170f.jpg";
    let currentImageBase64 = null;
    let isAnalyzing = false;
    let isFirstMessage = true;
    let currentDiseaseContext = null; //Track detected disease for RAG

    
    marked.setOptions({ breaks: true });
    feather.replace();

    // ui helpers
    modelSelectChat.addEventListener('change', () => {
        const isExpert = modelSelectChat.value === 'finetuned';
        currentModelBadge.textContent = isExpert ? 'Expert' : 'Base';
        currentModelBadge.className = isExpert 
            ? "text-[10px] font-bold text-primary-600 bg-primary-50 px-2 py-0.5 rounded-full uppercase tracking-wide border border-primary-100 inline-block mt-0.5"
            : "text-[10px] font-bold text-gray-500 bg-gray-100 px-2 py-0.5 rounded-full uppercase tracking-wide border border-gray-200 inline-block mt-0.5";
    });

    uploadBtn.addEventListener('click', () => imageInput.click());
    
    imageInput.addEventListener('change', function(e) {
        if (this.files && this.files[0]) {
            const reader = new FileReader();
            reader.onload = function(e) {
                currentImageBase64 = e.target.result;
                imagePreview.src = currentImageBase64;
                previewContainer.classList.remove('hidden');
                isFirstMessage = true;
                chatInput.focus();
            };
            reader.readAsDataURL(this.files[0]);
        }
    });

    clearImageBtn.addEventListener('click', () => {
        currentImageBase64 = null;
        imageInput.value = '';
        previewContainer.classList.add('hidden');
        currentDiseaseContext = null; // Reset disease tracking
        removeDiseaseBadge();
    });

    chatInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });

    newChatBtn.addEventListener('click', () => {
        chatContainer.style.opacity = '0';
        setTimeout(() => {
            currentDiseaseContext = null;
            removeDiseaseBadge();
            location.reload();
        }, 300);
    });

    // api & chat functions
    async function callApi(question, skipDescriptionBadge = false) {
        isAnalyzing = true;
        chatInput.disabled = true;
        sendBtn.disabled = true;
        
        // Show Loading Dots 
        const loadingId = showLoadingDots();
        scrollToBottom();

        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image: currentImageBase64,
                    question: question,
                    model_type: modelSelectChat.value,
                    is_first_turn: isFirstMessage,
                    disease_context: currentDiseaseContext // send disease context to RAG
                })
            });
            const data = await response.json();
            
            // rm loading
            const loadingEl = document.getElementById(loadingId);
            if(loadingEl) loadingEl.remove();
            
            if (data.success) {
                // show response
                addAIMessage(data.answer);
                isFirstMessage = false;
                
                // extract disease for future RAG queries
                if (modelSelectChat.value === 'finetuned') {
                    extractAndTrackDisease(data.answer);
                }
            } else {
                addErrorMessage(`⚠️ Error: ${data.error}`);
            }
        } catch (error) {
            const loadingEl = document.getElementById(loadingId);
            if(loadingEl) loadingEl.remove();
            addErrorMessage(`⚠️ Connection Error: ${error.message}`);
        } finally {
            isAnalyzing = false;
            chatInput.disabled = false;
            sendBtn.disabled = false;
            chatInput.focus();
            scrollToBottom();
        }
    }
    
    // get image description and show as badge
    async function getImageDescriptionBadge() {
        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image: currentImageBase64,
                    question: 'Describe this picture',
                    model_type: modelSelectChat.value,
                    is_first_turn: true,
                    disease_context: null
                })
            });
            const data = await response.json();
            
            if (data.success) {
                showDescriptionBadge(data.answer);
            }
        } catch (error) {
            console.warn('Failed to get image description:', error);
        }
    }
    
    // Show description badge (similar to disease badge)
    function showDescriptionBadge(description) {
        // Remove old badge if exists
        const oldBadge = document.getElementById('descriptionBadge');
        if (oldBadge) oldBadge.remove();
        
        const badge = document.createElement('div');
        badge.id = 'descriptionBadge';
        badge.className = 'fixed top-24 left-8 bg-gradient-to-r from-blue-500 to-purple-500 text-white px-4 py-2 rounded-2xl shadow-lg text-sm max-w-md message-slide z-50';
        badge.innerHTML = `
            <div class=\"flex items-start gap-2\">
                <svg class=\"w-4 h-4 mt-0.5 flex-shrink-0\" fill=\"currentColor\" viewBox=\"0 0 20 20\">
                    <path d=\"M10 12a2 2 0 100-4 2 2 0 000 4z\"/>
                    <path fill-rule=\"evenodd\" d=\"M.458 10C1.732 5.943 5.522 3 10 3s8.268 2.943 9.542 7c-1.274 4.057-5.064 7-9.542 7S1.732 14.057.458 10zM14 10a4 4 0 11-8 0 4 4 0 018 0z\" clip-rule=\"evenodd\"/>
                </svg>
                <div class=\"flex-1\">
                    <div class=\"font-semibold mb-1\">Image Analysis:</div>
                    <div class=\"text-xs opacity-90 line-clamp-3\">${description}</div>
                </div>
            </div>
        `;
        document.body.appendChild(badge);
    }

    // extract disease using base Qwen model
    async function extractAndTrackDisease(responseText) {
        try {
            const response = await fetch('/api/extract_disease', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ response_text: responseText })
            });
            
            const data = await response.json();
            
            if (data.success && data.disease) {
                currentDiseaseContext = data.disease;
                console.log(`✅ Disease detected: ${data.disease}`);
                showDiseaseBadge(data.disease);
            }
        } catch (error) {
            console.warn('Disease extraction failed:', error);
        }
    }

    // Show disease badge in UI
    function showDiseaseBadge(disease) {
        removeDiseaseBadge(); // remove old badge first
        
        const badge = document.createElement('div');
        badge.id = 'diseaseBadge';
        badge.className = 'fixed top-24 right-8 bg-gradient-to-r from-primary-500 to-rose-500 text-white px-4 py-2 rounded-full shadow-lg text-sm font-semibold flex items-center gap-2 message-slide z-50';
        badge.innerHTML = `
            <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clip-rule="evenodd"/>
            </svg>
            <span>Tracking: ${disease}</span>
        `;
        document.body.appendChild(badge);
    }

    function removeDiseaseBadge() {
        const badge = document.getElementById('diseaseBadge');
        if (badge) badge.remove();
    }

    sendBtn.addEventListener('click', sendMessage);
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    function sendMessage() {
        const message = chatInput.value.trim();
        if (!currentImageBase64) {
            const box = document.querySelector('.group .flex');
            box.classList.add('ring-2', 'ring-red-400');
            setTimeout(() => box.classList.remove('ring-2', 'ring-red-400'), 300);
            return;
        }
        if (!message || isAnalyzing) return;
        
        if(welcomeScreen) welcomeScreen.style.display = 'none';
        addUserMessage(message, isFirstMessage ? currentImageBase64 : null);
        
        chatInput.value = '';
        chatInput.style.height = 'auto';
        previewContainer.classList.add('hidden');
        callApi(message);
    }

    // renderers
    function addUserMessage(text, imgSrc) {
        const div = document.createElement('div');
        div.className = 'flex justify-end mb-6 message-slide';
        let imgHtml = imgSrc ? `<div class="mb-2"><img src="${imgSrc}" class="rounded-xl max-h-56 border-2 border-white shadow-lg"></div>` : '';
        div.innerHTML = `
            <div class="flex flex-col items-end max-w-[85%]">
                <div class="bg-gradient-to-br from-primary-600 to-primary-500 text-white rounded-2xl rounded-tr-sm px-5 py-3.5 shadow-xl">
                    ${imgHtml}
                    <p class="leading-relaxed text-[15px] whitespace-pre-wrap">${text}</p>
                </div>
            </div>
        `;
        chatContainer.appendChild(div);
        scrollToBottom();
    }

    // Direct Markdown Rendering - No Typing Effect
    function addAIMessage(text) {
        const div = document.createElement('div');
        div.className = 'flex gap-4 items-start mb-6 message-slide group';
        
        // Add disease indicator if tracking
        const diseaseTag = currentDiseaseContext ? `
            <div class="mb-2 inline-flex">
                <span class="inline-flex items-center gap-1.5 px-2.5 py-1 bg-primary-50 text-primary-700 rounded-full text-xs font-semibold border border-primary-200">
                    <svg class="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                        <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clip-rule="evenodd"/>
                    </svg>
                    ${currentDiseaseContext}
                </span>
            </div>
        ` : '';
        
        div.innerHTML = `
            <div class="w-9 h-9 rounded-full overflow-hidden flex-shrink-0 mt-1 border border-gray-100 shadow-md bg-white p-0.5">
                <img src="${BOT_AVATAR}" class="w-full h-full object-cover rounded-full">
            </div>
            <div class="flex-1">
                <div class="bg-white/90 backdrop-blur-sm text-gray-800 border border-gray-100 rounded-2xl rounded-tl-sm px-6 py-4 shadow-md inline-block message-content">
                    ${diseaseTag}
                    ${marked.parse(text)}
                </div>
                <div class="flex items-center gap-2 mt-2 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button onclick="copyToClipboard(this)" class="text-xs text-gray-400 hover:text-primary-600 flex items-center gap-1 px-2 py-1 rounded hover:bg-gray-50 transition-all">
                        <i data-feather="copy" class="w-3 h-3"></i>
                        <span>Copy</span>
                    </button>
                </div>
            </div>
        `;
        chatContainer.appendChild(div);
        feather.replace();
        scrollToBottom();
    }

    // Copy to clipboard function
    window.copyToClipboard = function(button) {
        const messageDiv = button.closest('.group').querySelector('.message-content');
        const text = messageDiv.innerText;
        
        navigator.clipboard.writeText(text).then(() => {
            const span = button.querySelector('span');
            const originalText = span.textContent;
            span.textContent = 'Copied!';
            button.classList.add('text-green-600');
            
            setTimeout(() => {
                span.textContent = originalText;
                button.classList.remove('text-green-600');
            }, 2000);
        });
    };

    function showLoadingDots() {
        const id = 'loading-' + Date.now();
        const div = document.createElement('div');
        div.id = id;
        div.className = 'flex gap-4 items-start mb-6 message-slide';
        div.innerHTML = `
            <div class="w-9 h-9 rounded-full overflow-hidden flex-shrink-0 mt-1 border border-gray-100 shadow-md bg-white p-0.5">
                <img src="${BOT_AVATAR}" class="w-full h-full object-cover rounded-full">
            </div>
            <div class="bg-white border border-gray-100 rounded-2xl rounded-tl-sm px-4 py-4 shadow-md flex items-center gap-2">
                <div class="flex gap-1">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
                <span class="text-xs text-gray-500">Analyzing...</span>
            </div>
        `;
        chatContainer.appendChild(div);
        scrollToBottom();
        return id;
    }

    function addErrorMessage(text) {
        const div = document.createElement('div');
        div.className = 'flex gap-4 items-start mb-6 animate-fade-in';
        div.innerHTML = `<div class="bg-red-50 text-red-600 border border-red-100 rounded-2xl px-5 py-3 ml-12 shadow-sm max-w-[90%] text-sm font-medium">${text}</div>`;
        chatContainer.appendChild(div);
        scrollToBottom();
    }

    function scrollToBottom() {
        chatContainer.scrollTo({ top: chatContainer.scrollHeight, behavior: 'smooth' });
    }

    // drag & drop image upload
    const dropZone = document.querySelector('main');
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.add('ring-4', 'ring-primary-300', 'ring-opacity-50');
        });
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('ring-4', 'ring-primary-300', 'ring-opacity-50');
        });
    });
    
    dropZone.addEventListener('drop', function(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files && files[0]) {
            const reader = new FileReader();
            reader.onload = function(e) {
                currentImageBase64 = e.target.result;
                imagePreview.src = currentImageBase64;
                previewContainer.classList.remove('hidden');
                isFirstMessage = true;
                chatInput.focus();
            };
            reader.readAsDataURL(files[0]);
        }
    });
});