import React, { useState } from 'react';
import { Search } from 'lucide-react';

const TextSearch = ({ onSearch }) => {
  const [query, setQuery] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim()) {
      onSearch(query);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex gap-2 w-full max-w-md mx-auto">
      <input
        type="text"
        placeholder="Nhập tên sản phẩm..."
        className="flex-1 p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 shadow-sm"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      <button
        type="submit"
        className="bg-blue-600 hover:bg-blue-700 text-white p-3 rounded-lg flex items-center gap-2 transition-colors shadow-sm"
      >
        <Search size={20} />
        <span>Tìm</span>
      </button>
    </form>
  );
};

export default TextSearch;