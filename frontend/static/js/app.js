// Toggle dark mode
function toggleDarkMode() {
  document.body.classList.toggle('dark');
}

// File upload drag-and-drop
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-upload');

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', async (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  fileInput.files = e.dataTransfer.files;
  uploadFile(file);
});

function triggerUpload() {
  fileInput.click();
  fileInput.onchange = () => {
    const file = fileInput.files[0];
    uploadFile(file);
  };
}

async function uploadFile(file) {
  if (!file) return;

  const formData = new FormData();
  formData.append('file', file);

  const chatBox = document.getElementById('chat-box');
  const uploading = document.createElement('div');
  uploading.className = 'bot-message fade-in';
  uploading.innerHTML = `<div class="text">📤 Uploading "${file.name}"...</div>`;
  chatBox.appendChild(uploading);
  chatBox.scrollTop = chatBox.scrollHeight;

  try {
    const response = await fetch('/upload', { method: 'POST', body: formData });
    const result = await response.json();

    uploading.innerHTML = `<div class="text">✅ "${file.name}" uploaded successfully!</div>`;
  } catch (err) {
    uploading.innerHTML = `<div class="text">❌ Upload failed.</div>`;
  }

  chatBox.scrollTop = chatBox.scrollHeight;
}

// Animate sidebar and chat on load
window.addEventListener('DOMContentLoaded', () => {
  document.querySelector('.sidebar').classList.add('slide-in');
  document.querySelector('.chat-area').classList.add('fade-in');
});

// Chat functionality
async function sendMessage() {
  const input = document.getElementById('user-input');
  const chatBox = document.getElementById('chat-box');
  const message = input.value.trim();
  if (!message) return;

  const userBubble = document.createElement('div');
  userBubble.className = 'user-message fade-in';
  userBubble.innerHTML = `<div class="text">${message}</div>`;
  chatBox.appendChild(userBubble);
  input.value = '';
  chatBox.scrollTop = chatBox.scrollHeight;

  // Typing indicator
  const botBubble = document.createElement('div');
  botBubble.className = 'bot-message fade-in';
  const botText = document.createElement('div');
  botText.className = 'text typing';
  botText.textContent = '🤖 ...';
  botBubble.appendChild(botText);
  chatBox.appendChild(botBubble);
  chatBox.scrollTop = chatBox.scrollHeight;

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({ question: message })
    });
    const data = await res.json();

    botText.textContent = '';
    let index = 0;
    const reply = data.answer || 'Sorry, I couldn’t understand that.';

    const typingInterval = setInterval(() => {
      if (index < reply.length) {
        botText.textContent += reply.charAt(index++);
        chatBox.scrollTop = chatBox.scrollHeight;
      } else {
        clearInterval(typingInterval);
      }
    }, 30);
  } catch (e) {
    botText.textContent = '❌ Failed to get response.';
  }
}
