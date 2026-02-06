import { useEffect, useRef, useState } from 'react';
import { Send, FileText, Image, File, Loader2 } from 'lucide-react';
import type { ChatMessage } from '../types';
import { api, ApiError } from '../services/api';
import { useDocuments } from '../contexts/DocumentContext';

const generateId = () => Math.random().toString(36).substring(2, 11);

const getFileIcon = (type: string) => {
  if (type.startsWith('image/')) return Image;
  if (type.includes('pdf') || type.includes('text')) return FileText;
  return File;
};

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const { documents, clearDocuments } = useDocuments();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    const handler = () => {
      setMessages([]);
      setInput('');
      clearDocuments();
      setStatus(null);
    };

    window.addEventListener('app:new-chat', handler);
    return () => window.removeEventListener('app:new-chat', handler);
  }, [clearDocuments]);

  useEffect(() => {
    if (textareaRef.current) {
      const textarea = textareaRef.current;
      textarea.style.height = 'auto';
      const scrollHeight = textarea.scrollHeight;
      const minHeight = 48;
      const maxHeight = 200;
      const newHeight = Math.min(Math.max(scrollHeight, minHeight), maxHeight);
      textarea.style.height = `${newHeight}px`;
      textarea.style.overflowY = scrollHeight > maxHeight ? 'auto' : 'hidden';
    }
  }, [input]);

  const handleSubmit = async () => {
    if (!input.trim() && documents.length === 0) return;

    const userMessage: ChatMessage = {
      id: generateId(),
      role: 'user',
      content: input,
      attachments: documents.length > 0 ? [...documents] : undefined,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    const currentInput = input;
    setInput('');
    setIsTyping(true);
    setStatus(null);

    try {
      const response = await api.query.advanced({ query: currentInput.trim(), stream: false });

      const assistantMessage: ChatMessage = {
        id: generateId(),
        role: 'assistant',
        content: response.response,
        citations: response.citations,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      const message = err instanceof ApiError ? err.message : 'Failed to get a response. Please try again.';
      setMessages((prev) => [
        ...prev,
        {
          id: generateId(),
          role: 'assistant',
          content: message,
          timestamp: new Date(),
        },
      ]);
    } finally {
      setIsTyping(false);
      setStatus(null);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="chat-screen">
      <div className="messages-area">
        {messages.length === 0 && !isTyping ? (
          <div className="chat-hero">
            <div className="logo-bubble">KG</div>
            <h1>Knowledge Graph</h1>
            <p>Connect your documents. Discover relationships. Get answers powered by graph-based reasoning.</p>
          </div>
        ) : (
          <div className="messages-scroll">
            {messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))}

            {isTyping && (
              <div className="typing-row animate-in">
                <div className="avatar ai">AI</div>
                <div className="typing-dots">
                  <span />
                  <span />
                  <span />
                </div>
              </div>
            )}

            {status && (
              <div className="status-banner">{status}</div>
            )}

            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      <div className="composer">
        <div className="composer-row">
          <div className="input-wrap">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Send a message..."
              rows={1}
              className="message-input"
            />
          </div>

          <button
            onClick={handleSubmit}
            disabled={(!input.trim() && documents.length === 0) || isTyping}
            className="send-btn"
          >
            {isTyping ? <Loader2 className="icon spinner" /> : <Send className="icon" />}
          </button>
        </div>

        <p className="input-hint">Enter to send • Shift+Enter for a new line</p>
      </div>
    </div>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === 'user';

  return (
    <div className={`message-row ${isUser ? 'is-user' : ''} animate-in`}>
      <div className={`avatar ${isUser ? 'user' : 'ai'}`}>{isUser ? 'U' : 'AI'}</div>

      <div className={`bubble-stack ${isUser ? 'align-end' : ''}`}>
        {message.attachments && message.attachments.length > 0 && (
          <div className="attachment-stack">
            {message.attachments.map((attachment) => {
              const FileIcon = getFileIcon(attachment.type);
              return (
                <div key={attachment.id} className="attachment-pill">
                  <FileIcon className="icon" />
                  <span>{attachment.name}</span>
                </div>
              );
            })}
          </div>
        )}

        {message.content && (
          <div className={`bubble ${isUser ? 'user' : 'ai'}`}>
            <MessageContent content={message.content} />
          </div>
        )}

        {message.citations && message.citations.length > 0 && (
          <div className="citation-row">
            {message.citations.map((citation, i) => (
              <span key={citation.reference_id} className="citation-pill" title={citation.content}>
                <FileText className="icon" />
                <span>{citation.source || `Source ${i + 1}`}</span>
              </span>
            ))}
          </div>
        )}

        <span className="timestamp">
          {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </span>
      </div>
    </div>
  );
}

function MessageContent({ content }: { content: string }) {
  const lines = content.split('\n');

  return (
    <>
      {lines.map((line, i) => {
        const formattedLine = line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

        if (line.startsWith('- ') || line.startsWith('* ')) {
          return (
            <div key={i} className="flex gap-1.5 sm:gap-2 ml-1 sm:ml-2">
              <span className="text-[hsl(var(--muted-foreground))]">•</span>
              <span dangerouslySetInnerHTML={{ __html: formattedLine.substring(2) }} />
            </div>
          );
        }

        const numberedMatch = line.match(/^(\d+)\.\s(.+)$/);
        if (numberedMatch) {
          return (
            <div key={i} className="flex gap-1.5 sm:gap-2 ml-1 sm:ml-2">
              <span className="text-[hsl(var(--muted-foreground))] min-w-[1.25em] sm:min-w-[1.5em]">{numberedMatch[1]}.</span>
              <span dangerouslySetInnerHTML={{ __html: numberedMatch[2] }} />
            </div>
          );
        }

        if (!line.trim()) {
          return <div key={i} className="h-2 sm:h-3" />;
        }

        return (
          <p key={i} dangerouslySetInnerHTML={{ __html: formattedLine }} />
        );
      })}
    </>
  );
}
