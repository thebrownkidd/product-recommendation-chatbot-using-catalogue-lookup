import React, { useState } from 'react';
import axios from 'axios';
import Results from './Results';

const ImageSearch = () => {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [detectedCategories, setDetectedCategories] = useState([]);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!image) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', image);

    try {
      const response = await axios.post(
        `${process.env.REACT_APP_API_URL || 'http://localhost:8000'}/search/image`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        }
      );
      setResults(response.data);
      
      // Extract detected categories from the response
      const categoryMatch = response.data.enhanced_response.match(/detected categories: (.*?)\)/);
      if (categoryMatch && categoryMatch[1]) {
        setDetectedCategories(categoryMatch[1].split(', '));
      }
    } catch (err) {
      setError('Error searching with image. Please try again.');
      console.error('Image search error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="search-container">
      <h2>Image Search</h2>
      <form onSubmit={handleSubmit} className="search-form">
        <div className="image-upload-container">
          <input
            type="file"
            accept="image/*"
            onChange={handleImageChange}
            className="file-input"
            id="image-upload"
          />
          <label htmlFor="image-upload" className="file-input-label">
            Choose Image
          </label>
          {preview && (
            <div className="image-preview">
              <img src={preview} alt="Preview" />
            </div>
          )}
        </div>
        <button 
          type="submit" 
          className="search-button" 
          disabled={loading || !image}
        >
          {loading ? 'Analyzing Image...' : 'Search with Image'}
        </button>
      </form>

      {error && <div className="error-message">{error}</div>}
      
      {detectedCategories.length > 0 && (
        <div className="detected-categories">
          <h4>Detected Categories:</h4>
          <div className="category-tags">
            {detectedCategories.map((cat, index) => (
              <span key={index} className="category-tag">{cat}</span>
            ))}
          </div>
        </div>
      )}
      
      {results && <Results data={results} />}
    </div>
  );
};

export default ImageSearch;