import { useEffect, useRef, useState, useCallback } from 'react';
import { Send, FileText, Image, File, Loader2, X, Brain } from 'lucide-react';
import type { ChatMessage, Citation } from '../types';
import { api, ApiError } from '../services/api';
import ReasoningTraceModal from '../components/ReasoningTraceModal';
import { useDocuments } from '../contexts/DocumentContext';
import { useQueryParams } from '../contexts/QueryParamsContext';
import QueryParamsPanel from '../components/QueryParamsPanel';

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
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null);
  const [traceQueryId, setTraceQueryId] = useState<string | null>(null);

  const closeCitation = useCallback(() => setActiveCitation(null), []);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const { documents, clearDocuments } = useDocuments();
  const { params } = useQueryParams();

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
      const response = await api.query.advanced({ query: currentInput.trim(), ...params });

      const assistantMessage: ChatMessage = {
        id: generateId(),
        role: 'assistant',
        content: response.response,
        citations: response.citations,
        queryId: response.metadata?.query_id,
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
            <div className="logo-bubble">ALR</div>
            <h1>Agentic LightRag</h1>
            <p>Connect your documents. Discover relationships. Get answers powered by agentic graph-based reasoning.</p>
          </div>
        ) : (
          <div className="messages-scroll">
            {messages.map((message) => (
              <MessageBubble key={message.id} message={message} onCitationClick={setActiveCitation} onShowTrace={setTraceQueryId} />
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

        <QueryParamsPanel compact className="px-1" />

        <p className="input-hint">Enter to send • Shift+Enter for a new line</p>
      </div>

      {activeCitation && (
        <CitationModal citation={activeCitation} onClose={closeCitation} />
      )}

      {traceQueryId && (
        <ReasoningTraceModal queryId={traceQueryId} onClose={() => setTraceQueryId(null)} />
      )}
    </div>
  );
}

function MessageBubble({ message, onCitationClick, onShowTrace }: { message: ChatMessage; onCitationClick: (c: Citation) => void; onShowTrace: (queryId: string) => void }) {
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
              <span
                key={citation.reference_id}
                className="citation-pill"
                onClick={() => onCitationClick(citation)}
              >
                <FileText className="icon" />
                <span>{citation.source || `Source ${i + 1}`}</span>
              </span>
            ))}
          </div>
        )}

        <div className="flex items-center gap-2">
          <span className="timestamp">
            {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </span>
          {!isUser && message.queryId && (
            <button
              className="reasoning-trace-btn"
              onClick={() => onShowTrace(message.queryId!)}
            >
              <Brain className="icon" />
              Show reasoning trace
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function CitationModal({ citation, onClose }: { citation: Citation; onClose: () => void }) {
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [onClose]);

  return (
    <div className="citation-overlay" onClick={onClose}>
      <div className="citation-modal" onClick={(e) => e.stopPropagation()}>
        <div className="citation-modal-header">
          <div className="citation-modal-title">
            <FileText className="icon" />
            <span>{citation.source || 'Source'}</span>
          </div>
          <button className="citation-modal-close" onClick={onClose}>
            <X className="icon" />
          </button>
        </div>
        <div className="citation-modal-body">
          {citation.content ? (
            <p>{citation.content}</p>
          ) : (
            <p className="text-[hsl(var(--muted-foreground))] italic">No content available</p>
          )}
          {citation.file_path && (
            <span className="citation-modal-path">{citation.file_path}</span>
          )}
        </div>
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
