import React, { useState, useEffect, useRef } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

const ChatbotComponent = ({ title = "Textbook Assistant" }) => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = { id: Date.now(), text: inputValue, isUser: true };
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // In a real implementation, this would call the backend API
      // For now, we'll simulate a response
      const response = await simulateChatResponse(inputValue);
      
      const botMessage = { 
        id: Date.now() + 1, 
        text: response.answer, 
        sources: response.sourceContent || [],
        isUser: false 
      };
      
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      const errorMessage = { 
        id: Date.now() + 1, 
        text: "Sorry, I encountered an error. Please try again.", 
        isUser: false 
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Simulate chat response (in real implementation, this would call the backend API)
  const simulateChatResponse = async (question) => {
    // This is a placeholder - in a real app, you would call your backend API here
    // Example API call:
    /*
    const response = await fetch('http://localhost:8000/v1/question', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question: question,
        selectedText: null, // Could be populated if user selects text
        userId: null // Could be populated if user is logged in
      })
    });
    
    return await response.json();
    */
    
    // For demonstration purposes, return a simulated response
    return new Promise(resolve => {
      setTimeout(() => {
        resolve({
          answer: `This is a simulated response to your question: "${question}". In a real implementation, this would come from the textbook RAG chatbot API.`,
          sourceContent: [
            { title: "Sample Source", section: "/docs/sample", relevance: 0.9 }
          ]
        });
      }, 1000);
    });
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="chatbot-container">
      <div className="chatbot-header">
        <h3>{title}</h3>
      </div>
      
      <div className="chatbot-messages">
        {messages.length === 0 ? (
          <div className="chatbot-welcome">
            <p>Hello! I'm your textbook assistant. Ask me anything about the content in this textbook.</p>
          </div>
        ) : (
          messages.map((message) => (
            <div 
              key={message.id} 
              className={`chatbot-message ${message.isUser ? 'user-message' : 'bot-message'}`}
            >
              <div className="message-content">
                {message.text}
              </div>
              {message.sources && message.sources.length > 0 && (
                <div className="message-sources">
                  <small>Sources:</small>
                  <ul>
                    {message.sources.map((source, idx) => (
                      <li key={idx}>
                        <a href={source.section} target="_blank" rel="noopener noreferrer">
                          {source.title}
                        </a> (Relevance: {(source.relevance * 100).toFixed(0)}%)
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ))
        )}
        {isLoading && (
          <div className="chatbot-message bot-message">
            <div className="message-content">
              <em>Thinking...</em>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      
      <div className="chatbot-input">
        <textarea
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about the textbook content..."
          rows="2"
          disabled={isLoading}
        />
        <button 
          onClick={handleSendMessage} 
          disabled={isLoading || !inputValue.trim()}
          className="send-button"
        >
          Send
        </button>
      </div>
    </div>
  );
};

// CSS styles for the chatbot
const ChatbotStyles = () => {
  return (
    <style>{`
      .chatbot-container {
        border: 1px solid #ddd;
        border-radius: 8px;
        height: 500px;
        display: flex;
        flex-direction: column;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
      }
      
      .chatbot-header {
        background-color: var(--ifm-color-primary);
        color: white;
        padding: 12px 16px;
        font-weight: bold;
      }
      
      .chatbot-messages {
        flex: 1;
        overflow-y: auto;
        padding: 16px;
        background-color: #fafafa;
      }
      
      .chatbot-welcome {
        color: #666;
        font-style: italic;
        text-align: center;
        padding: 20px;
      }
      
      .chatbot-message {
        margin-bottom: 16px;
        max-width: 80%;
      }
      
      .user-message {
        margin-left: auto;
        text-align: right;
      }
      
      .bot-message {
        margin-right: auto;
      }
      
      .message-content {
        padding: 12px 16px;
        border-radius: 18px;
        display: inline-block;
      }
      
      .user-message .message-content {
        background-color: var(--ifm-color-primary);
        color: white;
      }
      
      .bot-message .message-content {
        background-color: white;
        border: 1px solid #e0e0e0;
      }
      
      .message-sources {
        margin-top: 8px;
        padding: 8px;
        background-color: rgba(0,0,0,0.03);
        border-radius: 8px;
        font-size: 0.85em;
      }
      
      .message-sources ul {
        margin-bottom: 0;
        padding-left: 20px;
      }
      
      .message-sources li {
        margin-bottom: 4px;
      }
      
      .message-sources a {
        color: var(--ifm-color-primary);
      }
      
      .chatbot-input {
        padding: 16px;
        border-top: 1px solid #ddd;
        background-color: white;
        display: flex;
      }
      
      .chatbot-input textarea {
        flex: 1;
        padding: 12px;
        border: 1px solid #ddd;
        border-radius: 4px;
        resize: none;
        margin-right: 8px;
      }
      
      .send-button {
        padding: 12px 20px;
        background-color: var(--ifm-color-primary);
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
      }
      
      .send-button:disabled {
        background-color: #cccccc;
        cursor: not-allowed;
      }
    `}</style>
  );
};

const Chatbot = (props) => {
  return (
    <BrowserOnly>
      {() => {
        return (
          <>
            <ChatbotStyles />
            <ChatbotComponent {...props} />
          </>
        );
      }}
    </BrowserOnly>
  );
};

export default Chatbot;