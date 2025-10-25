import { useState, useEffect } from 'react';
import axios from 'axios';
import { Send, Zap, Database, TrendingUp, Trash2, List } from 'lucide-react';
import './App.css';

const API_URL = 'http://localhost:8000';

function App() {
  const [query, setQuery] = useState('');
  const [responses, setResponses] = useState([]);
  const [loading, setLoading] = useState(false);
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    fetchMetrics();
  }, []);

  const fetchMetrics = async () => {
    try {
      const res = await axios.get(`${API_URL}/stats`);
      setMetrics(res.data);
    } catch (error) {
      console.error('Error fetching metrics:', error);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    const startTime = Date.now();

    try {
      const res = await axios.post(`${API_URL}/query`, { query });
      const endTime = Date.now();

      setResponses([
        {
          query,
          ...res.data,
          timestamp: new Date().toLocaleTimeString(),
          clientTime: endTime - startTime
        },
        ...responses
      ]);

      setQuery('');
      fetchMetrics();
    } catch (error) {
      console.error('Error:', error);
      alert('Error processing query. Make sure backend is running on http://localhost:8000');
    } finally {
      setLoading(false);
    }
  };

  const clearCache = async () => {
    if (!window.confirm('Are you sure you want to clear all cached data?')) {
      return;
    }

    try {
      await axios.delete(`${API_URL}/cache/clear`);
      alert('Cache cleared successfully!');
      fetchMetrics();
      setResponses([]);
    } catch (error) {
      console.error('Error clearing cache:', error);
      alert('Error clearing cache');
    }
  };

  return (
    <div className="App">
      <header className="header">
        <h2>ML-Powered Query Caching with Vector Similarity</h2>
      </header>

      {/* Metrics Dashboard */}
      {metrics && (
        <div className="metrics-grid">
          <div className="metric-card">
            <Database size={24} color="#667eea" />
            <div>
              <h3>{metrics.total_requests}</h3>
              <p>Total Queries</p>
            </div>
          </div>
          <div className="metric-card">
            <Zap size={24} color="#10b981" />
            <div>
              <h3>{metrics.cache_hit_rate}%</h3>
              <p>Cache Hit Rate</p>
            </div>
          </div>

          <div className="metric-card">
            <List size={24} color="#8b5cf6" />
            <div>
              <h3>{metrics.total_cached_queries}</h3>
              <p>Cached Queries</p>
            </div>
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="action-buttons">
        <button onClick={clearCache} className="clear-button">
          <Trash2 size={18} />
          Clear Cache
        </button>
      </div>

      {/* Query Input */}
      <form onSubmit={handleSubmit} className="query-form">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask anything... (e.g., What is machine learning?)"
          disabled={loading}
          className="query-input"
        />
        <button type="submit" disabled={loading} className="submit-button">
          {loading ? (
            <span className="loading-spinner">⏳</span>
          ) : (
            <Send size={20} />
          )}
        </button>
      </form>

      {/* Response History */}
      <div className="responses">
        {responses.length === 0 ? (
          <div className="empty-state">
            <Zap size={48} color="#667eea" />
            <h3>No queries yet</h3>
          </div>
        ) : (
          responses.map((item, idx) => (
            <div key={idx} className={`response-card ${item.from_cache ? 'cached' : 'fresh'}`}>
              <div className="response-header">
                <strong className="query-text">Q: {item.query}</strong>
                <span className={`badge ${item.from_cache ? 'badge-cached' : 'badge-fresh'}`}>
                  {item.from_cache ? '⚡ Cached' : '🌐 API Call'}
                </span>
              </div>
              <p className="response-text">{item.response}</p>
              <div className="response-meta">
                <span className="meta-item">⏱️ {item.response_time_ms}ms</span>
                {item.from_cache && item.similarity && (
                  <span className="meta-item similarity">
                    🎯 {(item.similarity * 100).toFixed(1)}% similar to: "{item.original_query}"
                  </span>
                )}
                <span className="meta-item">{item.timestamp}</span>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default App;