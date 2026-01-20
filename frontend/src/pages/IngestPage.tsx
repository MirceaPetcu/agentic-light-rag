import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, X, Loader2, Check, File } from 'lucide-react';
import { api } from '../services/api';

type Tab = 'text' | 'file';
type Status = 'idle' | 'loading' | 'success' | 'error';

export default function IngestPage() {
  const [activeTab, setActiveTab] = useState<Tab>('text');
  const [textContent, setTextContent] = useState('');
  const [docId, setDocId] = useState('');
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [status, setStatus] = useState<Status>('idle');
  const [message, setMessage] = useState('');

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setSelectedFiles((prev) => [...prev, ...acceptedFiles]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/*': ['.txt', '.md', '.json', '.csv'],
      'application/json': ['.json'],
    },
    disabled: status === 'loading',
  });

  const handleTextIngest = async () => {
    if (!textContent.trim()) return;

    setStatus('loading');
    setMessage('');

    try {
      const response = await api.ingest.text({
        content: textContent,
        doc_id: docId || undefined,
      });

      setStatus('success');
      setMessage(response.message || 'Document added to knowledge base');
      setTextContent('');
      setDocId('');
    } catch (error) {
      setStatus('error');
      setMessage(error instanceof Error ? error.message : 'Failed to ingest document');
    }
  };

  const handleFileIngest = async () => {
    if (selectedFiles.length === 0) return;

    setStatus('loading');
    setMessage('');

    try {
      const results = await Promise.all(selectedFiles.map((file) => api.ingest.file(file)));
      const successCount = results.filter((r) => r.status === 'success').length;

      setStatus('success');
      setMessage(`${successCount} of ${selectedFiles.length} files uploaded`);
      setSelectedFiles([]);
    } catch (error) {
      setStatus('error');
      setMessage(error instanceof Error ? error.message : 'Failed to upload files');
    }
  };

  const removeFile = (index: number) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold text-[hsl(var(--foreground))] mb-2">
          Add documents
        </h1>
        <p className="text-[hsl(var(--muted-foreground))]">
          Upload content to build your knowledge base
        </p>
      </div>

      <div className="space-y-6">
        <div className="flex gap-2">
          <button
            onClick={() => setActiveTab('text')}
            className={`px-3 py-1.5 text-sm rounded-md transition-colors ${
              activeTab === 'text'
                ? 'bg-[hsl(var(--foreground))] text-[hsl(var(--background))]'
                : 'text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--foreground))]'
            }`}
          >
            Paste text
          </button>
          <button
            onClick={() => setActiveTab('file')}
            className={`px-3 py-1.5 text-sm rounded-md transition-colors ${
              activeTab === 'file'
                ? 'bg-[hsl(var(--foreground))] text-[hsl(var(--background))]'
                : 'text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--foreground))]'
            }`}
          >
            Upload files
          </button>
        </div>

        {activeTab === 'text' && (
          <div className="space-y-4 animate-in">
            <div>
              <label className="block text-sm text-[hsl(var(--muted-foreground))] mb-2">
                Document ID (optional)
              </label>
              <input
                type="text"
                placeholder="Leave empty for auto-generated ID"
                value={docId}
                onChange={(e) => setDocId(e.target.value)}
                className="
                  w-full px-3 py-2 rounded-lg text-sm
                  bg-[hsl(var(--muted))]
                  text-[hsl(var(--foreground))]
                  placeholder:text-[hsl(var(--muted-foreground))]
                  border-0
                  focus:outline-none focus:ring-1 focus:ring-[hsl(var(--border))]
                "
              />
            </div>

            <div>
              <label className="block text-sm text-[hsl(var(--muted-foreground))] mb-2">
                Content
              </label>
              <textarea
                placeholder="Paste your document content here..."
                rows={10}
                value={textContent}
                onChange={(e) => setTextContent(e.target.value)}
                className="
                  w-full px-4 py-3 rounded-lg resize-none
                  bg-[hsl(var(--muted))]
                  text-[hsl(var(--foreground))]
                  placeholder:text-[hsl(var(--muted-foreground))]
                  border-0
                  focus:outline-none focus:ring-1 focus:ring-[hsl(var(--border))]
                "
              />
            </div>

            <div className="flex justify-end">
              <button
                onClick={handleTextIngest}
                disabled={!textContent.trim() || status === 'loading'}
                className="
                  inline-flex items-center gap-2 px-4 py-2 rounded-lg
                  bg-[hsl(var(--foreground))] text-[hsl(var(--background))]
                  text-sm font-medium
                  disabled:opacity-40 disabled:cursor-not-allowed
                  hover:opacity-90 transition-opacity
                "
              >
                {status === 'loading' ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Upload className="w-4 h-4" />
                    Add document
                  </>
                )}
              </button>
            </div>
          </div>
        )}

        {activeTab === 'file' && (
          <div className="space-y-4 animate-in">
            <div
              {...getRootProps()}
              className={`
                border border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
                ${
                  isDragActive
                    ? 'border-[hsl(var(--foreground))] bg-[hsl(var(--muted))]'
                    : 'border-[hsl(var(--border))] hover:border-[hsl(var(--muted-foreground))]'
                }
                ${status === 'loading' ? 'opacity-50 cursor-not-allowed' : ''}
              `}
            >
              <input {...getInputProps()} />
              <Upload className="w-8 h-8 mx-auto mb-3 text-[hsl(var(--muted-foreground))]" />
              <p className="text-[hsl(var(--foreground))] mb-1">
                {isDragActive ? 'Drop files here' : 'Drop files or click to browse'}
              </p>
              <p className="text-sm text-[hsl(var(--muted-foreground))]">
                Supports .txt, .md, .json, .csv
              </p>
            </div>

            {selectedFiles.length > 0 && (
              <div className="space-y-2">
                {selectedFiles.map((file, index) => (
                  <div
                    key={`${file.name}-${index}`}
                    className="flex items-center justify-between p-3 rounded-lg bg-[hsl(var(--muted))]"
                  >
                    <div className="flex items-center gap-3 min-w-0">
                      <File className="w-4 h-4 text-[hsl(var(--muted-foreground))] flex-shrink-0" />
                      <div className="min-w-0">
                        <p className="text-sm text-[hsl(var(--foreground))] truncate">
                          {file.name}
                        </p>
                        <p className="text-xs text-[hsl(var(--muted-foreground))]">
                          {formatSize(file.size)}
                        </p>
                      </div>
                    </div>
                    <button
                      onClick={() => removeFile(index)}
                      className="p-1 rounded hover:bg-[hsl(var(--background))] transition-colors"
                    >
                      <X className="w-4 h-4 text-[hsl(var(--muted-foreground))]" />
                    </button>
                  </div>
                ))}
              </div>
            )}

            <div className="flex justify-end">
              <button
                onClick={handleFileIngest}
                disabled={selectedFiles.length === 0 || status === 'loading'}
                className="
                  inline-flex items-center gap-2 px-4 py-2 rounded-lg
                  bg-[hsl(var(--foreground))] text-[hsl(var(--background))]
                  text-sm font-medium
                  disabled:opacity-40 disabled:cursor-not-allowed
                  hover:opacity-90 transition-opacity
                "
              >
                {status === 'loading' ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Uploading...
                  </>
                ) : (
                  <>
                    <Upload className="w-4 h-4" />
                    Upload {selectedFiles.length > 0 ? `${selectedFiles.length} file${selectedFiles.length > 1 ? 's' : ''}` : ''}
                  </>
                )}
              </button>
            </div>
          </div>
        )}
      </div>

      {status !== 'idle' && status !== 'loading' && (
        <div
          className={`flex items-center gap-3 p-4 rounded-lg animate-in ${
            status === 'success'
              ? 'bg-[hsl(var(--success)/0.1)] text-[hsl(var(--success))]'
              : 'bg-[hsl(var(--destructive)/0.1)] text-[hsl(var(--destructive))]'
          }`}
        >
          {status === 'success' ? (
            <Check className="w-4 h-4" />
          ) : (
            <X className="w-4 h-4" />
          )}
          <span className="text-sm">{message}</span>
        </div>
      )}

      <div className="pt-4 border-t border-[hsl(var(--border))]">
        <p className="text-xs text-[hsl(var(--muted-foreground))]">
          Documents are processed into a knowledge graph for semantic retrieval.
          Large files may take longer to process.
        </p>
      </div>
    </div>
  );
}
