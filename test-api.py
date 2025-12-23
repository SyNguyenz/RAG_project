import requests
import json
from pathlib import Path

API_BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("\n" + "=" * 70)
    print("🏥 Testing Health Endpoint")
    print("=" * 70)
    
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_search_text(query: str = "red apple", k: int = 5):
    """Test text search"""
    print("\n" + "=" * 70)
    print(f"🔍 Testing Text Search: '{query}'")
    print("=" * 70)
    
    response = requests.get(
        f"{API_BASE_URL}/search/text",
        params={"query": query, "k": k}
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nQuery Type: {data['query_type']}")
        print(f"Total Results: {data['total_results']}")
        print(f"Search Time: {data['search_time_ms']:.2f}ms")
        
        print(f"\nTop {min(3, len(data['products']))} Results:")
        for i, product in enumerate(data['products'][:3], 1):
            print(f"\n{i}. {product['product_name']}")
            print(f"   Category: {product['category']}")
            print(f"   Price: {product['price']:,} {product['currency']}")
            print(f"   Similarity: {product['similarity_score']:.4f}")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200

def test_search_image(image_path: str, k: int = 5):
    """Test image search"""
    print("\n" + "=" * 70)
    print(f"🖼️  Testing Image Search: {image_path}")
    print("=" * 70)
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return False
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(
            f"{API_BASE_URL}/search/image",
            files=files,
            params={"k": k}
        )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nQuery Type: {data['query_type']}")
        print(f"Total Results: {data['total_results']}")
        print(f"Search Time: {data['search_time_ms']:.2f}ms")
        
        print(f"\nTop {min(3, len(data['products']))} Results:")
        for i, product in enumerate(data['products'][:3], 1):
            print(f"\n{i}. {product['product_name']}")
            print(f"   Category: {product['category']}")
            print(f"   Price: {product['price']:,} {product['currency']}")
            print(f"   Similarity: {product['similarity_score']:.4f}")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200

def test_get_product(product_id: str = "train_00000"):
    """Test get product by ID"""
    print("\n" + "=" * 70)
    print(f"📦 Testing Get Product: {product_id}")
    print("=" * 70)
    
    response = requests.get(f"{API_BASE_URL}/products/{product_id}")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        product = response.json()
        print(f"\nProduct Details:")
        print(f"Name: {product['product_name']}")
        print(f"Category: {product['category']}")
        print(f"Price: {product['price']:,} {product['currency']}")
        print(f"Description: {product['description']}")
        print(f"In Stock: {product['in_stock']}")
        print(f"Rating: {product['rating']}/5.0 ({product['reviews_count']} reviews)")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200

def test_list_products():
    """Test list products"""
    print("\n" + "=" * 70)
    print("📋 Testing List Products")
    print("=" * 70)
    
    response = requests.get(
        f"{API_BASE_URL}/products",
        params={"skip": 0, "limit": 5}
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        products = response.json()
        print(f"\nReturned {len(products)} products:")
        for i, product in enumerate(products, 1):
            print(f"{i}. {product['product_name']} - {product['price']:,} {product['currency']}")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200

def test_categories():
    """Test get categories"""
    print("\n" + "=" * 70)
    print("🗂️  Testing Get Categories")
    print("=" * 70)
    
    response = requests.get(f"{API_BASE_URL}/categories")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nTotal Categories: {data['total_categories']}")
        print(f"\nTop 10 Categories:")
        for cat in data['categories'][:10]:
            print(f"  - {cat['name']}: {cat['count']} products")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200

def test_stats():
    """Test get stats"""
    print("\n" + "=" * 70)
    print("📊 Testing Get Statistics")
    print("=" * 70)
    
    response = requests.get(f"{API_BASE_URL}/stats")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        stats = response.json()
        print(f"\nSystem Statistics:")
        print(f"Total Products: {stats['total_products']}")
        print(f"Total Categories: {stats['total_categories']}")
        print(f"In Stock: {stats['in_stock']}")
        print(f"Out of Stock: {stats['out_of_stock']}")
        print(f"\nPrice Range:")
        print(f"  Min: {stats['price_stats']['min']:,} VNĐ")
        print(f"  Max: {stats['price_stats']['max']:,} VNĐ")
        print(f"  Avg: {stats['price_stats']['avg']:,.0f} VNĐ")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200

def run_all_tests():
    """Chạy tất cả tests"""
    print("\n" + "=" * 70)
    print("🧪 RUNNING ALL API TESTS")
    print("=" * 70)
    
    results = {
        "Health Check": test_health(),
        "Text Search": test_search_text("apple"),
        "Get Product": test_get_product(),
        "List Products": test_list_products(),
        "Get Categories": test_categories(),
        "Get Statistics": test_stats(),
    }
    
    # Test image search nếu có ảnh mẫu
    sample_image = "./images/train/train_00000_label19_cattle.png"
    if Path(sample_image).exists():
        results["Image Search"] = test_search_image(sample_image)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\n{passed}/{total} tests passed ({passed/total*100:.1f}%)")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Chạy test cụ thể
        test_name = sys.argv[1]
        
        if test_name == "health":
            test_health()
        elif test_name == "text":
            query = sys.argv[2] if len(sys.argv) > 2 else "apple"
            test_search_text(query)
        elif test_name == "image":
            image_path = sys.argv[2] if len(sys.argv) > 2 else "./images/train/train_00000_label19_cattle.png"
            test_search_image(image_path)
        elif test_name == "product":
            product_id = sys.argv[2] if len(sys.argv) > 2 else "train_00000"
            test_get_product(product_id)
        elif test_name == "list":
            test_list_products()
        elif test_name == "categories":
            test_categories()
        elif test_name == "stats":
            test_stats()
        else:
            print(f"Unknown test: {test_name}")
    else:
        # Chạy tất cả tests
        run_all_tests()