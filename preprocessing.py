import numpy as np
import pickle
import json
import os
from PIL import Image
import random

def load_cifar100(data_dir='./cifar-100-python'):
    """
    Tải dữ liệu CIFAR-100
    """
    # Load training data
    train_file = os.path.join(data_dir, 'train')
    with open(train_file, 'rb') as f:
        train_dict = pickle.load(f, encoding='bytes')
    
    # Load test data
    test_file = os.path.join(data_dir, 'test')
    with open(test_file, 'rb') as f:
        test_dict = pickle.load(f, encoding='bytes')
    
    # Load metadata (fine labels)
    meta_file = os.path.join(data_dir, 'meta')
    with open(meta_file, 'rb') as f:
        meta_dict = pickle.load(f, encoding='bytes')
    
    return train_dict, test_dict, meta_dict

def save_images(data_dict, output_dir, prefix='train'):
    """
    Lưu ảnh từ CIFAR-100 ra file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    images = data_dict[b'data']
    fine_labels = data_dict[b'fine_labels']
    filenames = data_dict[b'filenames']
    
    # Reshape images (CIFAR-100: 32x32x3)
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    image_paths = []
    for idx, (img, label, filename) in enumerate(zip(images, fine_labels, filenames)):
        # Tạo tên file
        if isinstance(filename, bytes):
            filename = filename.decode('utf-8')
        
        img_filename = f"{prefix}_{idx:05d}_label{label}_{filename}"
        img_path = os.path.join(output_dir, img_filename)
        
        # Lưu ảnh
        img_pil = Image.fromarray(img)
        img_pil.save(img_path)
        
        image_paths.append({
            'id': f"{prefix}_{idx:05d}",
            'path': img_path,
            'label': int(label),
            'original_filename': filename
        })
    
    return image_paths

def create_product_metadata(image_info_list, label_names, output_file='product_metadata.json'):
    """
    Tạo metadata cho sản phẩm dựa trên CIFAR-100
    """
    # Danh sách mô tả mẫu
    descriptions_templates = [
        "Sản phẩm chất lượng cao, được nhiều khách hàng tin dùng",
        "Thiết kế hiện đại, phù hợp với mọi lứa tuổi",
        "Đặc biệt phù hợp cho việc sử dụng hàng ngày",
        "Sản phẩm độc quyền, có một không hai",
        "Được làm từ nguyên liệu cao cấp, bền đẹp",
        "Phong cách trẻ trung, năng động",
        "Thiết kế tinh tế, sang trọng",
        "Phù hợp làm quà tặng cho người thân"
    ]
    
    products = []
    
    for info in image_info_list:
        label_idx = info['label']
        label_name = label_names[label_idx].decode('utf-8') if isinstance(label_names[label_idx], bytes) else label_names[label_idx]
        
        # Tạo tên sản phẩm
        product_name = f"{label_name.title()} #{info['id'].split('_')[-1]}"
        
        # Giá ngẫu nhiên từ 50,000 đến 5,000,000 VNĐ
        price = random.randint(50, 5000) * 1000
        
        # Mô tả ngẫu nhiên
        description = random.choice(descriptions_templates)
        
        product = {
            'product_id': info['id'],
            'image_path': info['path'],
            'product_name': product_name,
            'category': label_name,
            'category_id': label_idx,
            'description': description,
            'price': price,
            'currency': 'VND',
            'in_stock': random.choice([True, True, True, False]),  # 75% có hàng
            'rating': round(random.uniform(3.5, 5.0), 1),
            'reviews_count': random.randint(0, 500)
        }
        
        products.append(product)
    
    # Lưu ra file JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(products, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Đã tạo metadata cho {len(products)} sản phẩm tại: {output_file}")
    return products

def create_category_mapping(label_names, coarse_labels, output_file='category_mapping.json'):
    """
    Tạo mapping giữa fine labels và coarse labels
    """
    fine_to_coarse = {}
    
    for fine_idx, coarse_idx in enumerate(coarse_labels):
        fine_name = label_names[fine_idx].decode('utf-8') if isinstance(label_names[fine_idx], bytes) else label_names[fine_idx]
        
        fine_to_coarse[fine_idx] = {
            'fine_label': fine_name,
            'fine_label_id': fine_idx,
            'coarse_label_id': int(coarse_idx)
        }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(fine_to_coarse, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Đã tạo category mapping tại: {output_file}")
    return fine_to_coarse

def main():
    """
    Hàm chính để preprocessing CIFAR-100
    """
    print("🚀 Bắt đầu preprocessing CIFAR-100...")
    
    # 1. Tải dữ liệu CIFAR-100
    print("\n📥 Đang tải CIFAR-100...")
    train_dict, test_dict, meta_dict = load_cifar100('./cifar-100-python')
    
    fine_label_names = meta_dict[b'fine_label_names']
    coarse_label_names = meta_dict[b'coarse_label_names']
    
    print(f"   - Số lượng fine labels: {len(fine_label_names)}")
    print(f"   - Số lượng coarse labels: {len(coarse_label_names)}")
    print(f"   - Số ảnh train: {len(train_dict[b'fine_labels'])}")
    print(f"   - Số ảnh test: {len(test_dict[b'fine_labels'])}")
    
    # 2. Lưu ảnh ra file
    print("\n💾 Đang lưu ảnh...")
    train_image_info = save_images(train_dict, './images/train', prefix='train')
    test_image_info = save_images(test_dict, './images/test', prefix='test')
    
    print(f"   ✅ Đã lưu {len(train_image_info)} ảnh train")
    print(f"   ✅ Đã lưu {len(test_image_info)} ảnh test")
    
    # 3. Tạo metadata sản phẩm
    print("\n📋 Đang tạo metadata sản phẩm...")
    all_image_info = train_image_info + test_image_info
    products = create_product_metadata(all_image_info, fine_label_names, 'product_metadata.json')
    
    # 4. Tạo category mapping
    print("\n🗂️  Đang tạo category mapping...")
    
    # Load coarse labels cho train và test
    train_coarse = train_dict[b'coarse_labels']
    test_coarse = test_dict[b'coarse_labels']
    all_coarse = train_coarse + test_coarse
    
    # Tạo mapping từ fine label đến coarse label
    fine_to_coarse_mapping = {}
    for img_info, coarse_label in zip(all_image_info, all_coarse):
        fine_label = img_info['label']
        if fine_label not in fine_to_coarse_mapping:
            fine_to_coarse_mapping[fine_label] = coarse_label
    
    # Tạo file mapping
    category_data = {}
    for fine_idx in range(100):
        fine_name = fine_label_names[fine_idx].decode('utf-8') if isinstance(fine_label_names[fine_idx], bytes) else fine_label_names[fine_idx]
        coarse_idx = fine_to_coarse_mapping.get(fine_idx, 0)
        coarse_name = coarse_label_names[coarse_idx].decode('utf-8') if isinstance(coarse_label_names[coarse_idx], bytes) else coarse_label_names[coarse_idx]
        
        category_data[fine_idx] = {
            'fine_label': fine_name,
            'fine_label_id': fine_idx,
            'coarse_label': coarse_name,
            'coarse_label_id': int(coarse_idx)
        }
    
    with open('category_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(category_data, f, ensure_ascii=False, indent=2)
    
    # 5. Tạo thống kê
    print("\n📊 Thống kê:")
    print(f"   - Tổng số sản phẩm: {len(products)}")
    print(f"   - Số categories (fine): 100")
    print(f"   - Số categories (coarse): 20")
    print(f"   - Giá trung bình: {sum(p['price'] for p in products) / len(products):,.0f} VNĐ")
    
    # Thống kê theo category
    from collections import Counter
    category_counts = Counter(p['category'] for p in products)
    print(f"\n   Top 5 categories:")
    for cat, count in category_counts.most_common(5):
        print(f"      - {cat}: {count} sản phẩm")
    
    print("\n✨ Hoàn thành preprocessing!")
    print("\n📁 Các file đã tạo:")
    print("   - ./images/train/ : Ảnh training")
    print("   - ./images/test/  : Ảnh test")
    print("   - product_metadata.json : Metadata sản phẩm")
    print("   - category_mapping.json : Mapping categories")

if __name__ == "__main__":
    # Cài đặt seed để có thể reproduce
    random.seed(42)
    np.random.seed(42)
    
    main()