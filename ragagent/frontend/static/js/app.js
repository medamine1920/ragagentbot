document.addEventListener('DOMContentLoaded', () => {
  const chatBox = document.getElementById('chatBox');
  const userInput = document.getElementById('userInput');
  const sendBtn = document.getElementById('sendBtn');
  const fileUpload = document.getElementById('fileUpload');

  function addMessage(text, isUser = false, fromCache = false) {
      const msgDiv = document.createElement('div');
      msgDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
      msgDiv.textContent = text;
      
      if (!isUser && fromCache) {
          const cacheNote = document.createElement('div');
          cacheNote.className = 'cache-message';
          cacheNote.textContent = '(From cache)';
          msgDiv.appendChild(cacheNote);
      }
      
      chatBox.appendChild(msgDiv);
      chatBox.scrollTop = chatBox.scrollHeight;
  }

  async function sendMessage() {
      const text = userInput.value.trim();
      if (!text) return;
      
      addMessage(text, true);
      userInput.value = '';
      
      try {
          const response = await fetch('/api/chat', {
              method: 'POST',
              headers: {
                  'Content-Type': 'application/json'
              },
              body: JSON.stringify({ query: text })
          });
          
          const data = await response.json();
          addMessage(data.answer, false, data.from_cache);
      } catch (error) {
          addMessage("Error: Could not get response", false);
      }
  }

  async function uploadFile() {
      const file = fileUpload.files[0];
      if (!file) return;
      
      const formData = new FormData();
      formData.append('file', file);
      
      try {
          const response = await fetch('/api/upload', {
              method: 'POST',
              body: formData
          });
          
          if (response.ok) {
              addMessage(`Document ${file.name} uploaded successfully`, false);
          }
      } catch (error) {
          addMessage("Error uploading file", false);
      }
      
      fileUpload.value = '';
  }

  sendBtn.addEventListener('click', sendMessage);
  userInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') sendMessage();
  });
  fileUpload.addEventListener('change', uploadFile);
});