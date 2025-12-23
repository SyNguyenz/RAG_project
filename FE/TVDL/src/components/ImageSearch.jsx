import React, { useState } from 'react';
import { Camera, Upload } from 'lucide-react';

const ImageSearch = ({ onSearch }) => {
  const [preview, setPreview] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      // Tạo preview ảnh
      const objectUrl = URL.createObjectURL(file);
      setPreview(objectUrl);

      // Chuẩn bị FormData để gửi đi
      const formData = new FormData();
      formData.append('file', file);

      // Gọi hàm search từ component cha
      onSearch(formData);
    }
  };

  return (
    <div className="w-full max-w-md mx-auto">
      <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-dashed border-gray-300 rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100 transition-colors">
        <div className="flex flex-col items-center justify-center pt-5 pb-6">
          {preview ? (
            <img src={preview} alt="Preview" className="h-20 object-contain mb-2 rounded" />
          ) : (
            <>
              <Camera className="w-8 h-8 mb-2 text-gray-500" />
              <p className="text-sm text-gray-500"><span className="font-semibold">Bấm để tải ảnh lên</span></p>
              <p className="text-xs text-gray-400">hoặc kéo thả vào đây</p>
            </>
          )}
        </div>
        <input 
          type="file" 
          className="hidden" 
          accept="image/*" 
          onChange={handleFileChange} 
        />
      </label>
    </div>
  );
};

export default ImageSearch;