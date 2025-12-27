import React, { useState } from 'react';
import MultimodalSearchBar from './components/MultimodalSearchBar';
import ProductGrid from './components/ProductGrid';
import { ShoppingBag, Sparkles, Bot } from 'lucide-react'; // Import thêm icon Sparkles/Bot

function App() {
  const [products, setProducts] = useState([]);
  const [aiMessage, setAiMessage] = useState(''); // State mới lưu tin nhắn AI
  const [isLoading, setIsLoading] = useState(false);
  const [searchStatus, setSearchStatus] = useState('');

  // Hàm xử lý logic gọi API
  const handleSearch = async ({ text, file }) => {
    setIsLoading(true);
    setProducts([]);
    setAiMessage(''); // Reset tin nhắn AI trước khi tìm kiếm mới
    setSearchStatus('');

    try {
      let url = '';
      let options = {};
      const formData = new FormData();

      // TRƯỜNG HỢP 1: MULTIMODAL (CẢ ẢNH VÀ TEXT)
      if (file && text) {
        console.log("Mode: Multimodal Search");
        url = 'http://localhost:8000/search/multimodal';
        
        formData.append('file', file);
        formData.append('text', text);
        formData.append('k', 10);
        formData.append('weight', 0.6);

        options = {
          method: 'POST',
          body: formData
        };
      }
      // TRƯỜNG HỢP 2: CHỈ CÓ ẢNH
      else if (file) {
        url = 'http://localhost:8000/search/image';
        formData.append('file', file);
        formData.append('k', 10);
        options = { method: 'POST', body: formData };
      }
      // TRƯỜNG HỢP 3: CHỈ CÓ TEXT (GET)
      else if (text) {
        const params = new URLSearchParams({ query: text, k: 10 });
        url = `http://localhost:8000/search/text?${params.toString()}`;
        options = { method: 'GET' };
      }

      // GỌI API
      const response = await fetch(url, options);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      // Lấy danh sách sản phẩm
      const resultProducts = data.products || data || [];
      
      // --- CẬP NHẬT: Lấy tin nhắn AI (nếu có) ---
      if (data.ai_message) {
        setAiMessage(data.ai_message);
      }
      
      if (resultProducts.length === 0) {
        setSearchStatus('Không tìm thấy sản phẩm nào phù hợp.');
      }
      setProducts(resultProducts);

    } catch (error) {
      console.error("Lỗi khi gọi API:", error);
      setSearchStatus('Có lỗi xảy ra khi kết nối tới máy chủ.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 font-sans text-gray-900">
      {/* Header & Search Section */}
      <div className="bg-white border-b border-gray-200 shadow-sm sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-4 pt-6 pb-8">
          
          {/* Logo / Title */}
          <div className="flex items-center justify-center gap-2 mb-8">
            <div className="bg-blue-600 text-white p-2 rounded-lg">
              <ShoppingBag size={28} />
            </div>
            <h1 className="text-3xl font-extrabold tracking-tight text-gray-800">
              Fashion<span className="text-blue-600">AI</span> Search
            </h1>
          </div>

          {/* THANH TÌM KIẾM MỚI */}
          <MultimodalSearchBar onSearch={handleSearch} />
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-6xl mx-auto px-4 py-8">
        
        {/* Loading State */}
        {isLoading && (
          <div className="flex flex-col items-center justify-center py-20">
            <div className="w-12 h-12 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin"></div>
            <p className="mt-4 text-gray-500 font-medium animate-pulse">Đang phân tích dữ liệu...</p>
          </div>
        )}

        {/* Empty / Error State */}
        {!isLoading && searchStatus && products.length === 0 && (
          <div className="text-center py-12">
            <p className="text-gray-500 text-lg">{searchStatus}</p>
          </div>
        )}

        {/* --- KHU VỰC HIỂN THỊ KẾT QUẢ --- */}
        {!isLoading && products.length > 0 && (
          <div>
            {/* 1. HIỂN THỊ TIN NHẮN TƯ VẤN CỦA AI (MỚI) */}
            {aiMessage && (
              <div className="mb-8 p-6 bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-100 rounded-2xl shadow-sm relative overflow-hidden animate-fade-in-up">
                <div className="flex gap-4 items-start relative z-10">
                  <div className="bg-white p-2 rounded-full shadow-sm text-blue-600 mt-1">
                    <Sparkles size={24} />
                  </div>
                  <div className="flex-1">
                    <h3 className="font-bold text-gray-800 text-lg mb-1 flex items-center gap-2">
                      Tư vấn viên AI
                      <span className="text-xs font-normal text-blue-600 bg-blue-100 px-2 py-0.5 rounded-full">Gemini Powered</span>
                    </h3>
                    <p className="text-gray-700 leading-relaxed text-base">
                      {aiMessage}
                    </p>
                  </div>
                </div>
                {/* Decoration background */}
                <div className="absolute top-0 right-0 -mt-4 -mr-4 w-24 h-24 bg-blue-100 rounded-full opacity-50 blur-xl"></div>
              </div>
            )}

            {/* 2. HIỂN THỊ GRID SẢN PHẨM */}
            <h2 className="text-xl font-bold text-gray-800 mb-6 px-1 flex items-center gap-2">
              Kết quả tìm kiếm 
              <span className="text-gray-500 text-base font-normal">({products.length} sản phẩm)</span>
            </h2>
            <ProductGrid products={products} />
          </div>
        )}

        {/* Welcome State (Khi chưa tìm gì) */}
        {!isLoading && !searchStatus && products.length === 0 && (
          <div className="text-center py-20 opacity-60">
            <div className="mb-4 flex justify-center text-gray-300">
               <Bot size={64} />
            </div>
            <p className="text-gray-400 text-lg">Hãy nhập từ khóa hoặc tải ảnh lên để AI tư vấn cho bạn</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;