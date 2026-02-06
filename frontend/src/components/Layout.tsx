import { useCallback, useEffect, useMemo, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  MessageSquare,
  BookOpen,
  Upload,
  FileText,
  Image,
  File,
  X,
  Plus,
  Loader2,
  Check,
  Clock,
} from 'lucide-react';
import { useDocuments } from '../contexts/DocumentContext';
import { api } from '../services/api';
import type { ChatAttachment, IngestionStatus } from '../types';

interface LayoutProps {
  children: React.ReactNode;
}

const navItems = [
  { path: '/', label: 'Chat', icon: MessageSquare },
  { path: '/query', label: 'Queries', icon: BookOpen },
  { path: '/ingest', label: 'Ingest', icon: Upload },
];

const STATUS_POLL_INTERVAL = 30_000;

const generateId = () => Math.random().toString(36).substring(2, 11);

const formatFileSize = (bytes: number) => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

const getFileIcon = (type: string) => {
  if (type.startsWith('image/')) return Image;
  if (type.includes('pdf') || type.includes('text')) return FileText;
  return File;
};

export default function Layout({ children }: LayoutProps) {
  const navigate = useNavigate();
  const location = useLocation();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { documents, addDocuments, removeDocument, updateDocumentMeta } = useDocuments();
  const documentsRef = useRef<ChatAttachment[]>(documents);

  useEffect(() => {
    document.documentElement.classList.add('dark');
  }, []);

  useEffect(() => {
    documentsRef.current = documents;
  }, [documents]);

  const activePath = useMemo(() => location.pathname, [location.pathname]);
  
  const normalizeBackendStatus = (status: string): IngestionStatus => {
    const lowered = status.toLowerCase();
    if (lowered === 'uploaded') return 'uploaded';
    if (lowered === 'processing') return 'processing';
    if (lowered === 'completed') return 'completed';
    if (lowered === 'failed') return 'failed';
    return 'unknown';
  };
  
  const refreshDocumentStatuses = useCallback(async () => {
    const docIds = documentsRef.current
      .map((doc) => doc.docId)
      .filter((id): id is string => Boolean(id));
  
    if (docIds.length === 0) return;
  
    try {
      const response = await api.documents.status(docIds);
      response.documents.forEach((statusRecord) => {
        const matching = documentsRef.current.find(
          (doc) => doc.docId === statusRecord.document_id
        );
        if (!matching) return;
  
        const normalizedStatus = normalizeBackendStatus(statusRecord.document_status);
  
        if (
          matching.ingestionStatus !== normalizedStatus ||
          (!matching.filePath && statusRecord.file_path)
        ) {
          updateDocumentMeta(matching.id, {
            ingestionStatus: normalizedStatus,
            filePath: statusRecord.file_path ?? matching.filePath,
          });
        }
      });
    } catch (error) {
      console.error('Failed to refresh document statuses', error);
    }
  }, [updateDocumentMeta]);
  
  useEffect(() => {
    let cancelled = false;
  
    const poll = async () => {
      if (cancelled) return;
      await refreshDocumentStatuses();
    };
  
    poll();
    const intervalId = window.setInterval(poll, STATUS_POLL_INTERVAL);
    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [refreshDocumentStatuses]);

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    const newAttachments: ChatAttachment[] = files.map((file) => ({
      id: generateId(),
      name: file.name,
      size: file.size,
      type: file.type,
      file,
      filePath: file.name,
      ingestionStatus: 'uploading' as IngestionStatus,
    }));
    addDocuments(newAttachments);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }

    // Start ingestion for each document
    for (const attachment of newAttachments) {
      if (!attachment.file) continue;
      try {
        const response = await api.ingest.file(attachment.file);
        updateDocumentMeta(attachment.id, {
          docId: response.doc_id || undefined,
          trackId: response.track_id || undefined,
          ingestionStatus: 'uploaded',
        });
      } catch {
        updateDocumentMeta(attachment.id, { ingestionStatus: 'failed' });
      }
    }

    await refreshDocumentStatuses();
  };

  const getStatusIcon = (status: IngestionStatus) => {
    switch (status) {
      case 'uploaded':
        return <Clock className="icon status-icon status-pending" />;
      case 'uploading':
        return <Loader2 className="icon status-icon status-uploading" />;
      case 'processing':
        return <Loader2 className="icon status-icon status-uploading" />;
      case 'completed':
        return <Check className="icon status-icon status-done" />;
      case 'failed':
        return <X className="icon status-icon status-failed" />;
      default:
        return <Clock className="icon status-icon status-pending" />;
    }
  };

  const getStatusLabel = (status: IngestionStatus) => {
    switch (status) {
      case 'uploaded':
        return 'Uploaded';
      case 'uploading':
        return 'Uploading...';
      case 'processing':
        return 'Processing...';
      case 'completed':
        return 'Completed';
      case 'failed':
        return 'Failed';
      default:
        return 'Pending status';
    }
  };

  return (
    <div className="app-shell">
      <div className="nav-rail">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activePath === item.path;
          return (
            <button
              key={item.path}
              className={`nav-rail__item ${isActive ? 'is-active' : ''}`}
              onClick={() => navigate(item.path)}
              aria-label={item.label}
            >
              <Icon className="icon" />
              <span>{item.label}</span>
            </button>
          );
        })}

      </div>

      <aside className="sidebar">
        <div className="sidebar-header">
          <label>Documents</label>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            onChange={handleFileSelect}
            className="hidden"
            accept=".pdf,.txt,.md,.doc,.docx,.csv,.json"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            className="add-doc-btn"
            title="Add document"
          >
            <Plus className="icon" />
          </button>
        </div>

        <div className="sidebar-list">
          {documents.length === 0 ? (
            <div className="no-docs">
              <FileText className="icon" />
              <p>No documents attached</p>
              <button onClick={() => fileInputRef.current?.click()}>
                Add documents
              </button>
            </div>
          ) : (
            documents.map((doc) => {
              const DocIcon = getFileIcon(doc.type);
              return (
                <div key={doc.id} className="doc-item">
                  <DocIcon className="icon doc-icon" />
                  <div className="doc-info">
                    <span className="doc-name">{doc.name}</span>
                    <div className="doc-meta">
                      <span className="doc-size">{formatFileSize(doc.size)}</span>
                      <span className="doc-status-badge" data-status={doc.ingestionStatus}>
                        {getStatusIcon(doc.ingestionStatus)}
                        <span>{getStatusLabel(doc.ingestionStatus)}</span>
                      </span>
                    </div>
                  </div>
                  <button
                    onClick={() => removeDocument(doc.id)}
                    className="doc-remove"
                    aria-label={`Remove ${doc.name}`}
                  >
                    <X className="icon" />
                  </button>
                </div>
              );
            })
          )}
        </div>
      </aside>

      <main className="chat-area">
        <header className="top-bar">
          <div className="top-bar__spacer" />
        </header>

        <div className="chat-content">{children}</div>
      </main>
    </div>
  );
}
