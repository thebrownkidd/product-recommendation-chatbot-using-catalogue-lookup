import React from 'react';
import ProductCard from './ProductCard';

const Results = ({ data }) => {
  if (!data || !data.products || data.products.length === 0) {
    return <p>No products found.</p>;
  }

  return (
    <div className="results-container">
      <div className="enhanced-response">
        <h3>Recommendation Summary</h3>
        <p>{data.enhanced_response}</p>
      </div>
      
      <h3>Recommended Products</h3>
      <div className="products-grid">
        {data.products.map((product, index) => (
          <ProductCard key={index} product={product} />
        ))}
      </div>
    </div>
  );
};

export default Results;