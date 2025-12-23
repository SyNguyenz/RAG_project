import torch
import clip
import numpy as np
import json
import os
from PIL import Image
from tqdm import tqdm
import pickle

class CLIPFeatureExtractor:
    """
    Trích xuất feature vectors từ ảnh sử dụng CLIP model
    """
    def __init__(self, model_name="ViT-B/32", device=None):
        """
        Args:
            model_name: Tên mô hình CLIP ("RN50", "ViT-B/32", "ViT-B/16", "ViT-L/14")
            device: Device để chạy model (cuda/cpu)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Đang tải mô hình CLIP {model_name} trên {self.device}...")
        
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        
        print(f"✅ Đã tải mô hình CLIP thành công!")
        print(f"   - Model: {model_name}")
        print(f"   - Device: {self.device}")
        print(f"   - Feature dimension: {self.model.visual.output_dim}")
    
    def extract_features(self, image_path):
        """
        Trích xuất feature vector từ một ảnh
        
        Args:
            image_path: Đường dẫn đến ảnh
            
        Returns:
            Feature vector (numpy array)
        """
        try:
            # Load và preprocess ảnh
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Trích xuất features
            with torch.no_grad():
                features = self.model.encode_image(image_input)
                # Normalize features (quan trọng cho similarity search)
                features = features / features.norm(dim=-1, keepdim=True)
            
            return features.cpu().numpy().flatten()
        
        except Exception as e:
            print(f"❌ Lỗi khi xử lý ảnh {image_path}: {e}")
            return None
    
    def extract_batch_features(self, image_paths, batch_size=32):
        """
        Trích xuất features cho nhiều ảnh (batch processing)
        
        Args:
            image_paths: List đường dẫn ảnh
            batch_size: Số ảnh xử lý cùng lúc
            
        Returns:
            List of feature vectors
        """
        all_features = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            valid_indices = []
            
            # Load và preprocess batch
            for idx, path in enumerate(batch_paths):
                try:
                    image = Image.open(path).convert('RGB')
                    image_input = self.preprocess(image)
                    batch_images.append(image_input)
                    valid_indices.append(idx)
                except Exception as e:
                    print(f"⚠️ Bỏ qua ảnh {path}: {e}")
                    all_features.append(None)
            
            if len(batch_images) == 0:
                continue
            
            # Stack thành batch tensor
            batch_tensor = torch.stack(batch_images).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model.encode_image(batch_tensor)
                features = features / features.norm(dim=-1, keepdim=True)
                features_np = features.cpu().numpy()
            
            # Thêm vào kết quả
            feature_idx = 0
            for idx in range(len(batch_paths)):
                if idx in valid_indices:
                    all_features.append(features_np[feature_idx])
                    feature_idx += 1
        
        return all_features

def process_cifar100_features(metadata_file='product_metadata.json', 
                               output_dir='./features',
                               model_name="ViT-B/32",
                               batch_size=32):
    """
    Xử lý toàn bộ CIFAR-100 và trích xuất features
    
    Args:
        metadata_file: File JSON chứa metadata sản phẩm
        output_dir: Thư mục lưu features
        model_name: Tên mô hình CLIP
        batch_size: Batch size cho feature extraction
    """
    print("=" * 70)
    print("🚀 BẮT ĐẦU TRÍCH XUẤT FEATURES TỪ CIFAR-100")
    print("=" * 70)
    
    # 1. Load metadata
    print(f"\n📂 Đang đọc metadata từ {metadata_file}...")
    with open(metadata_file, 'r', encoding='utf-8') as f:
        products = json.load(f)
    print(f"   ✅ Đã load {len(products)} sản phẩm")
    
    # 2. Khởi tạo CLIP model
    extractor = CLIPFeatureExtractor(model_name=model_name)
    feature_dim = extractor.model.visual.output_dim
    
    # 3. Chuẩn bị dữ liệu
    image_paths = [p['image_path'] for p in products]
    product_ids = [p['product_id'] for p in products]
    
    # 4. Trích xuất features
    print(f"\n🔍 Đang trích xuất features cho {len(image_paths)} ảnh...")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Feature dimension: {feature_dim}")
    
    features = extractor.extract_batch_features(image_paths, batch_size=batch_size)
    
    # 5. Xử lý kết quả
    valid_features = []
    valid_product_ids = []
    valid_indices = []
    
    for idx, (feat, prod_id) in enumerate(zip(features, product_ids)):
        if feat is not None:
            valid_features.append(feat)
            valid_product_ids.append(prod_id)
            valid_indices.append(idx)
    
    features_array = np.array(valid_features, dtype=np.float32)
    
    print(f"\n✅ Hoàn thành trích xuất features!")
    print(f"   - Số features thành công: {len(valid_features)}/{len(products)}")
    print(f"   - Shape: {features_array.shape}")
    print(f"   - Memory size: {features_array.nbytes / 1024 / 1024:.2f} MB")
    
    # 6. Lưu features
    os.makedirs(output_dir, exist_ok=True)
    
    # Lưu features array
    features_file = os.path.join(output_dir, 'features.npy')
    np.save(features_file, features_array)
    print(f"\n💾 Đã lưu features vào: {features_file}")
    
    # Lưu product IDs mapping
    ids_file = os.path.join(output_dir, 'product_ids.pkl')
    with open(ids_file, 'wb') as f:
        pickle.dump(valid_product_ids, f)
    print(f"💾 Đã lưu product IDs vào: {ids_file}")
    
    # Lưu metadata cho features
    metadata = {
        'model_name': model_name,
        'feature_dim': feature_dim,
        'total_products': len(products),
        'valid_features': len(valid_features),
        'feature_shape': features_array.shape,
        'product_ids': valid_product_ids
    }
    
    metadata_file_out = os.path.join(output_dir, 'features_metadata.json')
    with open(metadata_file_out, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"💾 Đã lưu metadata vào: {metadata_file_out}")
    
    # 7. Tạo index mapping (product_id -> feature_index)
    id_to_index = {prod_id: idx for idx, prod_id in enumerate(valid_product_ids)}
    index_file = os.path.join(output_dir, 'id_to_index.pkl')
    with open(index_file, 'wb') as f:
        pickle.dump(id_to_index, f)
    print(f"💾 Đã lưu index mapping vào: {index_file}")
    
    # 8. Thống kê
    print("\n" + "=" * 70)
    print("📊 THỐNG KÊ")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Feature dimension: {feature_dim}")
    print(f"Total products: {len(products)}")
    print(f"Successfully extracted: {len(valid_features)}")
    print(f"Failed: {len(products) - len(valid_features)}")
    print(f"Feature matrix shape: {features_array.shape}")
    print(f"Memory usage: {features_array.nbytes / 1024 / 1024:.2f} MB")
    
    # Phân tích theo category
    category_stats = {}
    for idx in valid_indices:
        cat = products[idx]['category']
        category_stats[cat] = category_stats.get(cat, 0) + 1
    
    print(f"\n📈 Thống kê theo category (top 10):")
    sorted_cats = sorted(category_stats.items(), key=lambda x: x[1], reverse=True)[:10]
    for cat, count in sorted_cats:
        print(f"   - {cat}: {count} products")
    
    print("\n" + "=" * 70)
    print("✨ HOÀN THÀNH!")
    print("=" * 70)
    
    return features_array, valid_product_ids, id_to_index

def test_feature_similarity(features_file='./features/features.npy',
                           ids_file='./features/product_ids.pkl',
                           metadata_file='product_metadata.json',
                           test_product_id='train_00000'):
    """
    Test tính năng tìm kiếm sản phẩm tương tự
    
    Args:
        features_file: File chứa feature vectors
        ids_file: File chứa product IDs
        metadata_file: File metadata sản phẩm
        test_product_id: ID sản phẩm để test
    """
    print("\n" + "=" * 70)
    print("🧪 TEST TÌM KIẾM SẢN PHẨM TƯƠNG TỰ")
    print("=" * 70)
    
    # Load data
    print("\n📂 Đang load dữ liệu...")
    features = np.load(features_file)
    with open(ids_file, 'rb') as f:
        product_ids = pickle.load(f)
    with open(metadata_file, 'r', encoding='utf-8') as f:
        products = json.load(f)
    
    # Tạo mapping
    id_to_product = {p['product_id']: p for p in products}
    id_to_index = {prod_id: idx for idx, prod_id in enumerate(product_ids)}
    
    # Lấy feature của sản phẩm test
    if test_product_id not in id_to_index:
        print(f"❌ Không tìm thấy product_id: {test_product_id}")
        return
    
    test_idx = id_to_index[test_product_id]
    test_feature = features[test_idx]
    test_product = id_to_product[test_product_id]
    
    print(f"\n🔍 Sản phẩm test:")
    print(f"   - ID: {test_product['product_id']}")
    print(f"   - Name: {test_product['product_name']}")
    print(f"   - Category: {test_product['category']}")
    print(f"   - Price: {test_product['price']:,} VNĐ")
    
    # Tính similarity với tất cả sản phẩm khác
    print(f"\n🔎 Tính toán similarity với {len(features)} sản phẩm...")
    similarities = np.dot(features, test_feature)
    
    # Lấy top 10 sản phẩm tương tự (bỏ qua chính nó)
    top_k = 11  # Lấy 11 để bỏ chính nó
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    print(f"\n🏆 TOP 10 SẢN PHẨM TƯƠNG TỰ:")
    print("-" * 70)
    
    for rank, idx in enumerate(top_indices, 1):
        if idx == test_idx:
            continue  # Skip chính nó
        
        similar_id = product_ids[idx]
        similar_product = id_to_product[similar_id]
        similarity_score = similarities[idx]
        
        print(f"\n{rank}. {similar_product['product_name']}")
        print(f"   - Category: {similar_product['category']}")
        print(f"   - Price: {similar_product['price']:,} VNĐ")
        print(f"   - Similarity: {similarity_score:.4f}")
        print(f"   - Same category: {'✅' if similar_product['category'] == test_product['category'] else '❌'}")

def main():
    """
    Hàm main để chạy toàn bộ pipeline
    """
    # Cấu hình
    CONFIG = {
        'metadata_file': 'product_metadata.json',
        'output_dir': './features',
        'model_name': 'ViT-B/32',  # Có thể thay đổi: "RN50", "ViT-B/16", "ViT-L/14"
        'batch_size': 64,  # Tùy theo GPU memory
    }
    
    # 1. Trích xuất features
    features, product_ids, id_to_index = process_cifar100_features(**CONFIG)
    
    # 2. Test similarity search
    print("\n")
    test_feature_similarity(
        features_file=os.path.join(CONFIG['output_dir'], 'features.npy'),
        ids_file=os.path.join(CONFIG['output_dir'], 'product_ids.pkl'),
        metadata_file=CONFIG['metadata_file'],
        test_product_id=product_ids[0]  # Test với sản phẩm đầu tiên
    )

if __name__ == "__main__":
    main()