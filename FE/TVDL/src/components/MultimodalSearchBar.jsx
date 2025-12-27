import React, { useState, useRef } from 'react';
import { Search, Camera, X, Image as ImageIcon } from 'lucide-react';

const MultimodalSearchBar = ({ onSearch }) => {
  const [text, setText] = useState('');
  const [preview, setPreview] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const fileInputRef = useRef(null);

  // Xử lý khi chọn ảnh
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      const objectUrl = URL.createObjectURL(file);
      setPreview(objectUrl);
    }
  };

  // Xử lý xóa ảnh đã chọn
  const handleRemoveImage = () => {
    setSelectedFile(null);
    setPreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = ''; // Reset input file
    }
  };

  // Xử lý submit form
  const handleSubmit = (e) => {
    e.preventDefault();
    if (!text.trim() && !selectedFile) return;

    // Gửi cả text và file lên App.js xử lý
    onSearch({ text, file: selectedFile });
  };

  // Kích hoạt input file ẩn
  const triggerFileUpload = () => {
    fileInputRef.current.click();
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      <form 
        onSubmit={handleSubmit} 
        className="relative flex flex-col gap-2 bg-white p-2 rounded-2xl shadow-lg border border-gray-200 transition-shadow focus-within:shadow-xl focus-within:border-blue-300"
      >
        {/* Khu vực hiển thị Preview Ảnh (nếu có) */}
        {preview && (
          <div className="flex px-2 pt-2">
            <div className="relative inline-block">
              <img 
                src={preview} 
                alt="Preview" 
                className="h-20 w-auto rounded-lg object-cover border border-gray-200 shadow-sm"
              />
              <button
                type="button"
                onClick={handleRemoveImage}
                className="absolute -top-2 -right-2 bg-gray-800 text-white rounded-full p-1 hover:bg-red-500 transition-colors shadow-md"
              >
                <X size={14} />
              </button>
            </div>
          </div>
        )}

        {/* Khu vực Input chính */}
        <div className="flex items-center gap-2 px-2 pb-1">
          {/* Nút Upload Ảnh */}
          <button
            type="button"
            onClick={triggerFileUpload}
            className="p-2 text-gray-400 hover:text-blue-600 hover:bg-blue-50 rounded-full transition-colors"
            title="Tải ảnh lên"
          >
            <Camera size={24} />
          </button>
          
          {/* Input File Ẩn */}
          <input 
            type="file" 
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="image/*"
            className="hidden"
          />

          {/* Input Text */}
          <input
            type="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder={preview ? "Nhập thêm mô tả về ảnh..." : "Tìm kiếm sản phẩm (nhập text hoặc tải ảnh)..."}
            className="flex-1 py-3 text-gray-700 placeholder-gray-400 outline-none text-lg bg-transparent"
          />

          {/* Nút Submit */}
          <button
            type="submit"
            disabled={!text.trim() && !selectedFile}
            className={`p-3 rounded-xl flex items-center gap-2 font-medium transition-all duration-200 ${
              (!text.trim() && !selectedFile)
                ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                : 'bg-blue-600 text-white hover:bg-blue-700 shadow-md'
            }`}
          >
            <Search size={20} />
            <span className="hidden sm:inline">Tìm kiếm</span>
          </button>
        </div>
      </form>
      
      {/* Gợi ý nhỏ bên dưới */}
      <div className="mt-3 flex justify-center gap-4 text-xs text-gray-400">
        <span className="flex items-center gap-1"><ImageIcon size={12}/> Hỗ trợ tìm kiếm bằng ảnh</span>
        <span className="flex items-center gap-1"><Search size={12}/> Kết hợp Text + Ảnh (RAG)</span>
      </div>
    </div>
  );
};

export default MultimodalSearchBar;