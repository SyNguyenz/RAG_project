import React from 'react';
import { Star, AlertCircle } from 'lucide-react';

// Cấu hình URL backend của bạn (khớp với uvicorn host/port)
const API_BASE_URL = 'http://localhost:8000';

const ProductGrid = ({ products }) => {
  // Hàm xử lý URL ảnh
  const getImageUrl = (path) => {
    if (!path) return 'https://via.placeholder.com/300?text=No+Image';
    if (path.startsWith('http')) return path;

    // 1. Xóa dấu / ở cuối API_BASE_URL (nếu lỡ tay khai báo thừa)
    const cleanBase = API_BASE_URL.replace(/\/+$/, '');
    
    // 2. Xóa dấu / ở đầu path (nếu có)
    const cleanPath = path.replace(/^\/+/, '');

    // 3. Nối lại với duy nhất 1 dấu / ở giữa
    return `${cleanBase}/${cleanPath}`;
  };

  // Kiểm tra nếu không có sản phẩm
  if (!products || products.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-gray-500 bg-white rounded-xl border border-dashed border-gray-300">
        <AlertCircle className="w-12 h-12 mb-3 text-gray-400" />
        <p className="text-lg">Không tìm thấy sản phẩm nào phù hợp.</p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6 mt-6 pb-10">
      {products.map((product) => (
        <div 
          key={product.id || product.product_id} // Fallback key
          className="group bg-white border border-gray-200 rounded-xl shadow-sm hover:shadow-xl transition-all duration-300 flex flex-col overflow-hidden relative"
        >
          {/* --- BADGES --- */}
          
          {/* Badge độ giống (Chỉ hiện khi search bằng ảnh) */}
          {product.similarity_score !== undefined && product.similarity_score !== null && (
            <div className="absolute top-2 right-2 z-10 bg-blue-600/90 backdrop-blur-sm text-white text-xs font-bold px-2 py-1 rounded-full shadow-sm">
              {(product.similarity_score * 100).toFixed(1)}% giống
            </div>
          )}

          {/* Badge hết hàng */}
          {!product.in_stock && (
            <div className="absolute top-2 left-2 z-10 bg-red-500 text-white text-xs font-bold px-2 py-1 rounded shadow-sm">
              Hết hàng
            </div>
          )}

          {/* --- IMAGE AREA --- */}
          <div className="relative h-56 overflow-hidden bg-gray-100">
            <img 
              src={getImageUrl(product.image_path)} 
              alt={product.product_name} 
              className={`w-full h-full object-cover transition-transform duration-500 group-hover:scale-105 ${!product.in_stock ? 'opacity-60 grayscale' : ''}`}
              onError={(e) => {
                e.target.src = 'https://via.placeholder.com/300?text=Image+Error'; 
                e.target.className = "w-full h-full object-contain p-4 bg-gray-50";
              }}
            />
          </div>

          {/* --- CONTENT AREA --- */}
          <div className="p-4 flex flex-col flex-grow">
            {/* Category */}
            <div className="flex justify-between items-center mb-1">
              <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
                {product.category || 'General'}
              </span>
              
              {/* Rating */}
              <div className="flex items-center gap-1 text-yellow-500">
                <Star size={12} fill="currentColor" />
                <span className="text-xs font-medium text-gray-700">
                  {product.rating} <span className="text-gray-400">({product.reviews_count})</span>
                </span>
              </div>
            </div>

            {/* Name */}
            <h3 className="text-gray-800 font-semibold text-lg leading-tight mb-2 line-clamp-2 min-h-[3rem]" title={product.product_name}>
              {product.product_name}
            </h3>

            {/* Description (Optional - ẩn trên màn hình nhỏ nếu cần gọn) */}
            <p className="text-sm text-gray-500 line-clamp-2 mb-4 h-10">
              {product.description}
            </p>

            {/* Footer: Price & Action */}
            <div className="mt-auto pt-3 border-t border-gray-100 flex items-center justify-between">
              <div className="flex flex-col">
                <span className="text-xs text-gray-400">Giá</span>
                <span className="text-lg font-bold text-red-600">
                  {product.price.toLocaleString('vi-VN')} {product.currency || '₫'}
                </span>
              </div>

              <button 
                disabled={!product.in_stock}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  product.in_stock 
                    ? 'bg-gray-900 text-white hover:bg-gray-800' 
                    : 'bg-gray-100 text-gray-400 cursor-not-allowed'
                }`}
              >
                {product.in_stock ? 'Chi tiết' : 'Liên hệ'}
              </button>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};

export default ProductGrid;