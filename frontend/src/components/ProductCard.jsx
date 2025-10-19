import React from 'react';

const ProductCard = ({ product }) => {
  return (
    <div className="product-card">
      <div className="product-image">
        {product.image_url ? (
          <img src={product.image_url} alt={product.title} />
        ) : (
          <div className="no-image">No Image</div>
        )}
      </div>
      <div className="product-info">
        <h3>{product.title}</h3>
        {product.price && product.price > 0 && (
          <p className="product-price">${product.price.toFixed(2)}</p>
        )}
        <p className="product-description">{product.description}</p>
        <div className="product-attributes">
          {Object.entries(product.attributes || {}).map(([key, value]) => (
            <span key={key} className="attribute">
              {key}: {value}
            </span>
          ))}
        </div>
        <div className="similarity-score">
          Similarity: {(product.similarity * 100).toFixed(1)}%
        </div>
      </div>
    </div>
  );
};

export default ProductCard;