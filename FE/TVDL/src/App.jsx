import React, { useState } from 'react';
import TextSearch from './components/TextSearch';
import ImageSearch from './components/ImageSearch';
import ProductGrid from './components/ProductGrid';
import axios from 'axios';

function App() {
  const [activeTab, setActiveTab] = useState('text'); // 'text' hoặc 'image'
  const [products, setProducts] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  // Giả lập hàm gọi API
  const mockApiCall = async (endpoint, payload) => {
    setIsLoading(true);
    console.log(`Đang gọi API: ${endpoint}`, payload);
    
    // Giả lập độ trễ mạng
    await new Promise(resolve => setTimeout(resolve, 1000));

    // Dữ liệu giả trả về
    const mockData = Array.from({ length: 8 }).map((_, i) => ({
      id: i,
      name: `Sản phẩm Demo ${i + 1}`,
      price: (Math.floor(Math.random() * 500) + 100) * 1000,
      imageUrl: `https://via.placeholder.com/300?text=Product+${i+1}`
    }));
    
    setProducts(mockData);
    setIsLoading(false);
  };

  // Xử lý tìm kiếm Text
  const handleTextSearch = (query) => {
    mockApiCall('/search/text', { query });
  };

  // Xử lý tìm kiếm Image
  const handleImageSearch = async (formData) => {
    //mockApiCall('/search/image', formData);
    try {
      const response = await fetch('http://localhost:8000/search/image?k=10', {
        method: 'POST',
        body: formData,
      });
      
      const data = await response.json();
      
      // QUAN TRỌNG: Truyền data.products vào state, không phải toàn bộ data
      setProducts(data.products); 
      
    } catch (error) {
      console.error("Lỗi:", error);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-10 px-4">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold text-center text-gray-800 mb-8">
          Hệ Thống Tìm Kiếm Thông Minh
        </h1>

        {/* Tab chuyển đổi */}
        <div className="flex justify-center mb-6">
          <div className="bg-white p-1 rounded-lg border border-gray-200 shadow-sm flex">
            <button
              className={`px-6 py-2 rounded-md transition-all ${
                activeTab === 'text' ? 'bg-blue-600 text-white shadow' : 'text-gray-600 hover:bg-gray-100'
              }`}
              onClick={() => setActiveTab('text')}
            >
              Tìm bằng Chữ
            </button>
            <button
              className={`px-6 py-2 rounded-md transition-all ${
                activeTab === 'image' ? 'bg-blue-600 text-white shadow' : 'text-gray-600 hover:bg-gray-100'
              }`}
              onClick={() => setActiveTab('image')}
            >
              Tìm bằng Ảnh
            </button>
          </div>
        </div>

        {/* Khu vực Input */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 mb-8">
          {activeTab === 'text' ? (
            <TextSearch onSearch={handleTextSearch} />
          ) : (
            <ImageSearch onSearch={handleImageSearch} />
          )}
        </div>

        {/* Khu vực Hiển thị kết quả */}
        {isLoading ? (
          <div className="text-center py-10">
            <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-600 mx-auto"></div>
            <p className="mt-4 text-gray-500">Đang tìm kiếm...</p>
          </div>
        ) : (
          <ProductGrid products={products} />
        )}
      </div>
    </div>
  );
}

export default App;