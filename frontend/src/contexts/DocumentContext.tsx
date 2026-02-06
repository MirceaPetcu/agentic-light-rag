import { createContext, useContext, useEffect, useState, type ReactNode } from 'react';
import type { ChatAttachment, IngestionStatus } from '../types';

const STORAGE_KEY = 'attached-documents';

const sanitizeForStorage = (docs: ChatAttachment[]) =>
  docs.map(({ file, ...rest }) => rest);

const loadDocuments = (): ChatAttachment[] => {
  if (typeof window === 'undefined') return [];
  const stored = localStorage.getItem(STORAGE_KEY);
  if (!stored) return [];

  try {
    const parsed = JSON.parse(stored);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
};

interface DocumentContextType {
  documents: ChatAttachment[];
  addDocuments: (docs: ChatAttachment[]) => void;
  removeDocument: (id: string) => void;
  clearDocuments: () => void;
  updateDocumentStatus: (id: string, status: IngestionStatus) => void;
  updateDocumentMeta: (
    id: string,
    meta: Partial<Pick<ChatAttachment, 'ingestionStatus' | 'docId' | 'trackId' | 'filePath'>>
  ) => void;
}

const DocumentContext = createContext<DocumentContextType | null>(null);

export function DocumentProvider({ children }: { children: ReactNode }) {
  const [documents, setDocuments] = useState<ChatAttachment[]>(() => loadDocuments());

  useEffect(() => {
    if (typeof window === 'undefined') return;

    if (documents.length === 0) {
      localStorage.removeItem(STORAGE_KEY);
    } else {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(sanitizeForStorage(documents)));
    }
  }, [documents]);

  const addDocuments = (docs: ChatAttachment[]) => {
    setDocuments((prev) => [...prev, ...docs]);
  };

  const removeDocument = (id: string) => {
    setDocuments((prev) => prev.filter((d) => d.id !== id));
  };

  const clearDocuments = () => {
    setDocuments([]);
  };

  const updateDocumentStatus = (id: string, status: IngestionStatus) => {
    setDocuments((prev) =>
      prev.map((doc) => (doc.id === id ? { ...doc, ingestionStatus: status } : doc))
    );
  };

  const updateDocumentMeta = (
    id: string,
    meta: Partial<Pick<ChatAttachment, 'ingestionStatus' | 'docId' | 'trackId' | 'filePath'>>
  ) => {
    setDocuments((prev) => prev.map((doc) => (doc.id === id ? { ...doc, ...meta } : doc)));
  };

  return (
    <DocumentContext.Provider
      value={{ documents, addDocuments, removeDocument, clearDocuments, updateDocumentStatus, updateDocumentMeta }}
    >
      {children}
    </DocumentContext.Provider>
  );
}

export function useDocuments() {
  const context = useContext(DocumentContext);
  if (!context) {
    throw new Error('useDocuments must be used within a DocumentProvider');
  }
  return context;
}
