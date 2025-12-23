import faiss
import numpy as np
import pickle
import json
import os
import time
from typing import List, Tuple, Dict

class FAISSIndexManager:
    """
    Quản lý FAISS index cho image retrieval
    """
    def __init__(self, feature_dim: int):
        """
        Args:
            feature_dim: Số chiều của feature vector
        """
        self.feature_dim = feature_dim
        self.index = None
        self.index_type = None
        
    def create_flat_index(self, features: np.ndarray, use_gpu: bool = False):
        """
        Tạo Flat index (exact search) - Tìm kiếm chính xác
        
        Args:
            features: Feature vectors [N, D]
            use_gpu: Có sử dụng GPU không
            
        Returns:
            FAISS index
        """
        print(f"\n🔨 Đang tạo IndexFlatL2...")
        print(f"   - Feature dimension: {self.feature_dim}")
        print(f"   - Number of vectors: {len(features)}")
        print(f"   - Use GPU: {use_gpu}")
        
        # Tạo index
        index = faiss.IndexFlatL2(self.feature_dim)
        
        # Chuyển sang GPU nếu cần
        if use_gpu and faiss.get_num_gpus() > 0:
            print(f"   - Chuyển index sang GPU...")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        
        # Thêm vectors vào index
        start_time = time.time()
        index.add(features.astype(np.float32))
        elapsed = time.time() - start_time
        
        print(f"   ✅ Hoàn thành trong {elapsed:.2f}s")
        print(f"   - Total vectors in index: {index.ntotal}")
        
        self.index = index
        self.index_type = "IndexFlatL2"
        return index
    
    def create_ivf_index(self, features: np.ndarray, 
                        nlist: int = 100, 
                        nprobe: int = 10,
                        use_gpu: bool = False):
        """
        Tạo IVF index (approximate search) - Tìm kiếm xấp xỉ nhanh hơn
        
        Args:
            features: Feature vectors [N, D]
            nlist: Số lượng clusters (càng lớn càng chính xác nhưng chậm hơn)
            nprobe: Số clusters tìm kiếm (càng lớn càng chính xác nhưng chậm hơn)
            use_gpu: Có sử dụng GPU không
            
        Returns:
            FAISS index
        """
        print(f"\n🔨 Đang tạo IndexIVFFlat...")
        print(f"   - Feature dimension: {self.feature_dim}")
        print(f"   - Number of vectors: {len(features)}")
        print(f"   - nlist (clusters): {nlist}")
        print(f"   - nprobe (search): {nprobe}")
        print(f"   - Use GPU: {use_gpu}")
        
        # Tạo quantizer
        quantizer = faiss.IndexFlatL2(self.feature_dim)
        
        # Tạo IVF index
        index = faiss.IndexIVFFlat(quantizer, self.feature_dim, nlist)
        
        # Chuyển sang GPU nếu cần
        if use_gpu and faiss.get_num_gpus() > 0:
            print(f"   - Chuyển index sang GPU...")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        
        # Train index với một phần dữ liệu
        print(f"   - Training index...")
        start_time = time.time()
        
        # Sample data cho training nếu quá lớn
        train_size = min(len(features), 10000)
        train_data = features[:train_size].astype(np.float32)
        index.train(train_data)
        
        train_time = time.time() - start_time
        print(f"   ✅ Training hoàn thành trong {train_time:.2f}s")
        
        # Thêm vectors vào index
        print(f"   - Adding vectors to index...")
        start_time = time.time()
        index.add(features.astype(np.float32))
        add_time = time.time() - start_time
        
        # Set nprobe
        index.nprobe = nprobe
        
        print(f"   ✅ Hoàn thành trong {add_time:.2f}s")
        print(f"   - Total vectors in index: {index.ntotal}")
        
        self.index = index
        self.index_type = f"IndexIVFFlat_nlist{nlist}_nprobe{nprobe}"
        return index
    
    def create_hnsw_index(self, features: np.ndarray, 
                         M: int = 32, 
                         efConstruction: int = 200,
                         efSearch: int = 64):
        """
        Tạo HNSW index (Hierarchical Navigable Small World) - Rất nhanh và chính xác
        
        Args:
            features: Feature vectors [N, D]
            M: Số lượng kết nối trên mỗi layer (16-64, mặc định 32)
            efConstruction: Độ rộng tìm kiếm khi xây dựng (100-500)
            efSearch: Độ rộng tìm kiếm khi query (10-500)
            
        Returns:
            FAISS index
        """
        print(f"\n🔨 Đang tạo IndexHNSWFlat...")
        print(f"   - Feature dimension: {self.feature_dim}")
        print(f"   - Number of vectors: {len(features)}")
        print(f"   - M: {M}")
        print(f"   - efConstruction: {efConstruction}")
        print(f"   - efSearch: {efSearch}")
        
        # Tạo HNSW index
        index = faiss.IndexHNSWFlat(self.feature_dim, M)
        index.hnsw.efConstruction = efConstruction
        index.hnsw.efSearch = efSearch
        
        # Thêm vectors
        print(f"   - Adding vectors...")
        start_time = time.time()
        index.add(features.astype(np.float32))
        elapsed = time.time() - start_time
        
        print(f"   ✅ Hoàn thành trong {elapsed:.2f}s")
        print(f"   - Total vectors in index: {index.ntotal}")
        
        self.index = index
        self.index_type = f"IndexHNSWFlat_M{M}_ef{efSearch}"
        return index
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tìm kiếm k vectors gần nhất
        
        Args:
            query_vector: Query vector [1, D] hoặc [D]
            k: Số lượng kết quả trả về
            
        Returns:
            distances, indices: Khoảng cách và indices của k vectors gần nhất
        """
        if self.index is None:
            raise ValueError("Index chưa được tạo!")
        
        # Reshape query vector nếu cần
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        query_vector = query_vector.astype(np.float32)
        
        # Search
        distances, indices = self.index.search(query_vector, k)
        
        return distances[0], indices[0]
    
    def save_index(self, filepath: str):
        """
        Lưu FAISS index ra file
        
        Args:
            filepath: Đường dẫn file .index
        """
        if self.index is None:
            raise ValueError("Index chưa được tạo!")
        
        # Convert GPU index về CPU trước khi lưu
        if hasattr(self.index, 'index'):  # GPU index
            cpu_index = faiss.index_gpu_to_cpu(self.index)
        else:
            cpu_index = self.index
        
        faiss.write_index(cpu_index, filepath)
        print(f"💾 Đã lưu index vào: {filepath}")
    
    def load_index(self, filepath: str, use_gpu: bool = False):
        """
        Load FAISS index từ file
        
        Args:
            filepath: Đường dẫn file .index
            use_gpu: Có chuyển lên GPU không
        """
        print(f"📂 Đang load index từ: {filepath}")
        index = faiss.read_index(filepath)
        
        if use_gpu and faiss.get_num_gpus() > 0:
            print(f"   - Chuyển index sang GPU...")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        
        self.index = index
        print(f"   ✅ Đã load {index.ntotal} vectors")
        
        return index

def build_faiss_indexes(features_dir: str = './features',
                       output_dir: str = './faiss_indexes',
                       build_all: bool = True):
    """
    Xây dựng các FAISS indexes
    
    Args:
        features_dir: Thư mục chứa features
        output_dir: Thư mục lưu indexes
        build_all: Xây dựng tất cả loại index
    """
    print("=" * 70)
    print("🏗️  XÂY DỰNG FAISS INDEXES")
    print("=" * 70)
    
    # 1. Load features
    print("\n📂 Đang load features...")
    features_file = os.path.join(features_dir, 'features.npy')
    metadata_file = os.path.join(features_dir, 'features_metadata.json')
    
    features = np.load(features_file)
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    print(f"   ✅ Đã load features")
    print(f"   - Shape: {features.shape}")
    print(f"   - Feature dim: {metadata['feature_dim']}")
    print(f"   - Total vectors: {len(features)}")
    
    # 2. Tạo output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. Khởi tạo manager
    manager = FAISSIndexManager(feature_dim=metadata['feature_dim'])
    
    # 4. Tạo các loại index
    indexes_info = []
    
    # 4.1. IndexFlatL2 (Exact Search)
    print("\n" + "=" * 70)
    print("1️⃣  INDEXFLATL2 - EXACT SEARCH")
    print("=" * 70)
    manager.create_flat_index(features, use_gpu=False)
    
    flat_index_file = os.path.join(output_dir, 'index_flat_l2.index')
    manager.save_index(flat_index_file)
    
    indexes_info.append({
        'type': 'IndexFlatL2',
        'file': flat_index_file,
        'description': 'Exact search - chính xác 100% nhưng chậm với dữ liệu lớn',
        'speed': 'Slow',
        'accuracy': '100%',
        'recommended_for': 'Dataset nhỏ (<100K vectors)'
    })
    
    if build_all:
        # 4.2. IndexIVFFlat (Approximate Search)
        print("\n" + "=" * 70)
        print("2️⃣  INDEXIVFFLAT - APPROXIMATE SEARCH")
        print("=" * 70)
        
        # Tính nlist dựa trên số lượng vectors
        n_vectors = len(features)
        nlist = min(int(np.sqrt(n_vectors)), 1000)  # Rule of thumb
        nprobe = max(int(nlist * 0.1), 10)  # 10% của nlist
        
        manager.create_ivf_index(features, nlist=nlist, nprobe=nprobe, use_gpu=False)
        
        ivf_index_file = os.path.join(output_dir, 'index_ivf_flat.index')
        manager.save_index(ivf_index_file)
        
        indexes_info.append({
            'type': 'IndexIVFFlat',
            'file': ivf_index_file,
            'nlist': nlist,
            'nprobe': nprobe,
            'description': 'Approximate search - nhanh hơn, độ chính xác ~95%',
            'speed': 'Medium-Fast',
            'accuracy': '~95%',
            'recommended_for': 'Dataset trung bình (100K-1M vectors)'
        })
        
        # 4.3. IndexHNSWFlat (Fast & Accurate)
        print("\n" + "=" * 70)
        print("3️⃣  INDEXHNSWFLAT - FAST & ACCURATE")
        print("=" * 70)
        manager.create_hnsw_index(features, M=32, efConstruction=200, efSearch=64)
        
        hnsw_index_file = os.path.join(output_dir, 'index_hnsw_flat.index')
        manager.save_index(hnsw_index_file)
        
        indexes_info.append({
            'type': 'IndexHNSWFlat',
            'file': hnsw_index_file,
            'M': 32,
            'efSearch': 64,
            'description': 'Hierarchical NSW - rất nhanh và chính xác ~99%',
            'speed': 'Very Fast',
            'accuracy': '~99%',
            'recommended_for': 'Production - tốt cho mọi kích thước dataset'
        })
    
    # 5. Lưu indexes info
    info_file = os.path.join(output_dir, 'indexes_info.json')
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(indexes_info, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 70)
    print("📊 TỔNG KẾT")
    print("=" * 70)
    print(f"Đã tạo {len(indexes_info)} FAISS indexes:")
    for i, info in enumerate(indexes_info, 1):
        print(f"\n{i}. {info['type']}")
        print(f"   - File: {info['file']}")
        print(f"   - Speed: {info['speed']}")
        print(f"   - Accuracy: {info['accuracy']}")
        print(f"   - Recommended: {info['recommended_for']}")
    
    print("\n" + "=" * 70)
    print("✨ HOÀN THÀNH!")
    print("=" * 70)
    
    return indexes_info

def benchmark_indexes(features_dir: str = './features',
                     indexes_dir: str = './faiss_indexes',
                     n_queries: int = 100,
                     k: int = 10):
    """
    So sánh hiệu suất của các indexes
    
    Args:
        features_dir: Thư mục chứa features
        indexes_dir: Thư mục chứa indexes
        n_queries: Số lượng queries để test
        k: Số kết quả trả về
    """
    print("\n" + "=" * 70)
    print("⚡ BENCHMARK FAISS INDEXES")
    print("=" * 70)
    
    # Load features
    features_file = os.path.join(features_dir, 'features.npy')
    features = np.load(features_file)
    
    # Chọn random queries
    np.random.seed(42)
    query_indices = np.random.choice(len(features), n_queries, replace=False)
    queries = features[query_indices]
    
    # Load indexes info
    info_file = os.path.join(indexes_dir, 'indexes_info.json')
    with open(info_file, 'r') as f:
        indexes_info = json.load(f)
    
    results = []
    
    for info in indexes_info:
        print(f"\n🔍 Testing {info['type']}...")
        
        # Load index
        manager = FAISSIndexManager(features.shape[1])
        manager.load_index(info['file'])
        
        # Benchmark
        start_time = time.time()
        
        for query in queries:
            distances, indices = manager.search(query, k=k)
        
        elapsed = time.time() - start_time
        avg_time = elapsed / n_queries * 1000  # ms
        qps = n_queries / elapsed  # queries per second
        
        result = {
            'type': info['type'],
            'total_time': f"{elapsed:.2f}s",
            'avg_query_time': f"{avg_time:.2f}ms",
            'qps': f"{qps:.2f}",
            'accuracy': info['accuracy']
        }
        results.append(result)
        
        print(f"   - Total time: {elapsed:.2f}s")
        print(f"   - Avg query time: {avg_time:.2f}ms")
        print(f"   - QPS: {qps:.2f}")
    
    # In bảng so sánh
    print("\n" + "=" * 70)
    print("📊 SO SÁNH HIỆU SUẤT")
    print("=" * 70)
    print(f"{'Index Type':<20} {'Avg Time':<12} {'QPS':<10} {'Accuracy':<10}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['type']:<20} {r['avg_query_time']:<12} {r['qps']:<10} {r['accuracy']:<10}")
    
    print("\n💡 Gợi ý:")
    print("   - IndexFlatL2: Dùng cho demo/test nhỏ")
    print("   - IndexIVFFlat: Dùng khi cần cân bằng tốc độ/độ chính xác")
    print("   - IndexHNSWFlat: Khuyến nghị cho production (nhanh + chính xác)")
    
    return results

def main():
    """
    Main function
    """
    # 1. Xây dựng indexes
    print("\n🚀 Bắt đầu xây dựng FAISS indexes...\n")
    
    indexes_info = build_faiss_indexes(
        features_dir='./features',
        output_dir='./faiss_indexes',
        build_all=True  # Xây dựng tất cả loại index
    )
    
    # 2. Benchmark
    print("\n\n🔬 Bắt đầu benchmark...\n")
    
    benchmark_results = benchmark_indexes(
        features_dir='./features',
        indexes_dir='./faiss_indexes',
        n_queries=100,
        k=10
    )
    
    print("\n✨ Hoàn thành tất cả!")

if __name__ == "__main__":
    main()