import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, File, X } from 'lucide-react';

interface FileDropzoneProps {
  onFilesSelected: (files: File[]) => void;
  selectedFiles: File[];
  onRemoveFile: (index: number) => void;
  accept?: Record<string, string[]>;
  maxFiles?: number;
  disabled?: boolean;
}

export default function FileDropzone({
  onFilesSelected,
  selectedFiles,
  onRemoveFile,
  accept = {
    'text/*': ['.txt', '.md', '.json', '.csv'],
    'application/json': ['.json'],
  },
  maxFiles = 10,
  disabled = false,
}: FileDropzoneProps) {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      onFilesSelected(acceptedFiles);
    },
    [onFilesSelected]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept,
    maxFiles,
    disabled,
  });

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="space-y-4">
      <div
        {...getRootProps()}
        className={`
          relative border-2 border-dashed rounded-xl p-8
          transition-all duration-200 cursor-pointer
          ${
            isDragActive
              ? 'border-[hsl(var(--primary))] bg-[hsl(var(--primary)/0.05)]'
              : 'border-[hsl(var(--border))] hover:border-[hsl(var(--primary)/0.5)] hover:bg-[hsl(var(--muted)/0.5)]'
          }
          ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
        `}
      >
        <input {...getInputProps()} />
        <div className="flex flex-col items-center text-center">
          <div
            className={`
            p-4 rounded-full mb-4 transition-colors
            ${isDragActive ? 'bg-[hsl(var(--primary)/0.1)]' : 'bg-[hsl(var(--muted))]'}
          `}
          >
            <Upload
              className={`w-8 h-8 ${isDragActive ? 'text-[hsl(var(--primary))]' : 'text-[hsl(var(--muted-foreground))]'}`}
            />
          </div>
          <p className="text-[hsl(var(--foreground))] font-medium mb-1">
            {isDragActive ? 'Drop files here' : 'Drag & drop files here'}
          </p>
          <p className="text-sm text-[hsl(var(--muted-foreground))]">
            or click to browse (txt, md, json, csv)
          </p>
        </div>
      </div>

      {selectedFiles.length > 0 && (
        <div className="space-y-2">
          <p className="text-sm font-medium text-[hsl(var(--foreground))]">
            Selected files ({selectedFiles.length})
          </p>
          <div className="space-y-2">
            {selectedFiles.map((file, index) => (
              <div
                key={`${file.name}-${index}`}
                className="flex items-center justify-between p-3 rounded-lg bg-[hsl(var(--muted))]"
              >
                <div className="flex items-center gap-3">
                  <File className="w-5 h-5 text-[hsl(var(--primary))]" />
                  <div>
                    <p className="text-sm font-medium text-[hsl(var(--foreground))] truncate max-w-[200px]">
                      {file.name}
                    </p>
                    <p className="text-xs text-[hsl(var(--muted-foreground))]">
                      {formatFileSize(file.size)}
                    </p>
                  </div>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onRemoveFile(index);
                  }}
                  className="p-1.5 rounded-md hover:bg-[hsl(var(--destructive)/0.1)] transition-colors"
                >
                  <X className="w-4 h-4 text-[hsl(var(--destructive))]" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
