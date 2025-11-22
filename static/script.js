// Chatbot Frontend JavaScript
// Handles user interactions and API communication

// DOM Elements
const chatContainer = document.getElementById('chatContainer');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');
const loadingOverlay = document.getElementById('loadingOverlay');
const statusIndicator = document.getElementById('statusIndicator');

// Initialize timestamp for welcome message
document.addEventListener('DOMContentLoaded', function () {
    const welcomeTime = document.getElementById('welcomeTime');
    if (welcomeTime) {
        welcomeTime.textContent = getCurrentTime();
    }
});

/**
 * Get current time in HH:MM format
 */
function getCurrentTime() {
    const now = new Date();
    return now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
}

/**
 * Show typing animation
 */
function showTypingAnimation() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot-message';
    typingDiv.id = 'typingIndicator';

    typingDiv.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;

    chatContainer.appendChild(typingDiv);
    scrollToBottom();
}

/**
 * Remove typing animation
 */
function removeTypingAnimation() {
    const typingIndicator = document.getElementById('typingIndicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

/**
 * Add user message to chat
 */
function addUserMessage(message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user-message';

    messageDiv.innerHTML = `
        <div class="message-avatar">👤</div>
        <div class="message-content">
            <p>${escapeHtml(message)}</p>
            <span class="timestamp">${getCurrentTime()}</span>
        </div>
    `;

    chatContainer.appendChild(messageDiv);
    scrollToBottom();
}

/**
 * Add bot message to chat
 */
function addBotMessage(message, intent = null, confidence = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot-message';

    let extraInfo = '';
    if (intent && confidence !== null) {
        extraInfo = `<span class="timestamp" style="opacity: 0.6;">Intent: ${intent} (${(confidence * 100).toFixed(1)}%)</span>`;
    }

    messageDiv.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <p>${escapeHtml(message)}</p>
            <span class="timestamp">${getCurrentTime()}</span>
            ${extraInfo}
        </div>
    `;

    chatContainer.appendChild(messageDiv);
    scrollToBottom();
}

/**
 * Escape HTML to prevent XSS attacks
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Scroll chat to bottom
 */
function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

/**
 * Send message to backend
 */
async function sendMessage() {
    const message = userInput.value.trim();

    if (!message) return;

    // Disable input while processing
    userInput.disabled = true;
    sendButton.disabled = true;

    // Add user message to chat
    addUserMessage(message);

    // Clear input
    userInput.value = '';

    // Show typing animation
    showTypingAnimation();

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message })
        });

        const data = await response.json();

        removeTypingAnimation();

        if (data.response) {
            await typeMessage(data.response, data.intent, data.confidence);
        } else {
            addBotMessage("I'm sorry, I couldn't process your request. Please try again.");
        }

    } catch (error) {
        console.error('Error:', error);
        removeTypingAnimation();
        addBotMessage("I'm sorry, I encountered an error. Please check your connection and try again.");
    } finally {
        userInput.disabled = false;
        sendButton.disabled = false;
        userInput.focus();
    }
}

/**
 * Type message with unique ID (FIXED)
 */
async function typeMessage(message, intent, confidence) {
    const uniqueId = "typingText_" + Date.now();

    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot-message';

    messageDiv.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <p id="${uniqueId}"></p>
            <span class="timestamp">${getCurrentTime()}</span>
            ${intent ? `<span class="timestamp" style="opacity: 0.6;">Intent: ${intent} (${(confidence * 100).toFixed(1)}%)</span>` : ''}
        </div>
    `;

    chatContainer.appendChild(messageDiv);
    const typingTextElement = document.getElementById(uniqueId);

    for (let i = 0; i <= message.length; i++) {
        typingTextElement.textContent = message.substring(0, i);
        await sleep(20);
        scrollToBottom();
    }
}

/**
 * Sleep function for delays
 */
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Event Listeners
 */
sendButton.addEventListener('click', sendMessage);

userInput.addEventListener('keypress', function (event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
});

/**
 * Connection Health Check
 */
async function checkConnection() {
    try {
        const response = await fetch('/health');
        const data = await response.json();

        if (data.status === 'healthy') {
            statusIndicator.innerHTML = `
                <span class="status-dot"></span>
                <span class="status-text">Online</span>
            `;
        } else {
            statusIndicator.innerHTML = `
                <span class="status-dot" style="background:#f44336;"></span>
                <span class="status-text">Offline</span>
            `;
        }
    } catch {
        statusIndicator.innerHTML = `
            <span class="status-dot" style="background:#f44336;"></span>
            <span class="status-text">Offline</span>
        `;
    }
}

checkConnection();
setInterval(checkConnection, 30000);
