import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import Layout from './components/Layout';
import { ChatPage, QueryPage, IngestPage } from './pages';
import { DocumentProvider } from './contexts/DocumentContext';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60,
      retry: 1,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <DocumentProvider>
        <BrowserRouter>
          <Layout>
            <Routes>
              <Route path="/" element={<ChatPage />} />
              <Route path="/query" element={<QueryPage />} />
              <Route path="/ingest" element={<IngestPage />} />
            </Routes>
          </Layout>
        </BrowserRouter>
      </DocumentProvider>
    </QueryClientProvider>
  );
}

export default App;
