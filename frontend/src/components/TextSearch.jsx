import React, { useState } from 'react';
import axios from 'axios';
import Results from './Results';

const TextSearch = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(`${process.env.REACT_APP_API_URL || 'http://localhost:8000'}/search/text`, {
        query: query
      });
      setResults(response.data);
    } catch (err) {
      setError('Error searching products. Please try again.');
      console.error('Search error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="search-container">
      <h2>Text Search</h2>
      <form onSubmit={handleSubmit} className="search-form">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="What are you looking for?"
          className="search-input"
        />
        <button type="submit" className="search-button" disabled={loading}>
          {loading ? 'Searching...' : 'Search'}
        </button>
      </form>

      {error && <div className="error-message">{error}</div>}
      
      {results && <Results data={results} />}
    </div>
  );
};

export default TextSearch;