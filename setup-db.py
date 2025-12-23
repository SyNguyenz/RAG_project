"""
PostgreSQL Database Setup Script
Tạo database và import dữ liệu từ product_metadata.json
"""

import psycopg2
from psycopg2.extras import execute_values
import json
import os
from typing import List, Dict
from dotenv import load_dotenv

# ============================================================================
# Load environments variables
# ============================================================================
load_dotenv()


# ============================================================================
# Database Configuration
# ============================================================================

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)), 
    'database': os.getenv('DB_NAME', 'image_retrieval_db'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres')
}

# ============================================================================
# SQL Schemas
# ============================================================================

CREATE_DATABASE_SQL = """
CREATE DATABASE image_retrieval_db;
"""

CREATE_PRODUCTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    product_id VARCHAR(50) UNIQUE NOT NULL,
    product_name VARCHAR(255) NOT NULL,
    category VARCHAR(100) NOT NULL,
    category_id INTEGER NOT NULL,
    description TEXT,
    price INTEGER NOT NULL,
    currency VARCHAR(10) DEFAULT 'VND',
    image_path TEXT NOT NULL,
    in_stock BOOLEAN DEFAULT TRUE,
    rating DECIMAL(2,1) DEFAULT 0.0,
    reviews_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_product_id ON products(product_id);",
    "CREATE INDEX IF NOT EXISTS idx_category ON products(category);",
    "CREATE INDEX IF NOT EXISTS idx_category_id ON products(category_id);",
    "CREATE INDEX IF NOT EXISTS idx_in_stock ON products(in_stock);",
    "CREATE INDEX IF NOT EXISTS idx_price ON products(price);",
    "CREATE INDEX IF NOT EXISTS idx_rating ON products(rating);",
]

# ============================================================================
# Database Functions
# ============================================================================

def create_database():
    """Tạo database (chỉ chạy lần đầu)"""
    print("\n🔧 Creating database...")
    
    try:
        # Connect to default postgres database
        conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            database='postgres',
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (DB_CONFIG['database'],)
        )
        
        if cursor.fetchone():
            print(f"   ℹ️  Database '{DB_CONFIG['database']}' already exists")
        else:
            cursor.execute(CREATE_DATABASE_SQL)
            print(f"   ✅ Database '{DB_CONFIG['database']}' created")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"   ⚠️  Note: {e}")
        print(f"   ℹ️  Continuing with existing database...")

def get_connection():
    """Kết nối đến database"""
    return psycopg2.connect(**DB_CONFIG)

def create_tables():
    """Tạo bảng products"""
    print("\n📋 Creating tables...")
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Create products table
        cursor.execute(CREATE_PRODUCTS_TABLE_SQL)
        print("   ✅ Table 'products' created")
        
        # Create indexes
        for sql in CREATE_INDEXES_SQL:
            cursor.execute(sql)
        print("   ✅ Indexes created")
        
        conn.commit()
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()

def load_products_from_json(json_file: str = 'product_metadata.json'):
    """Load dữ liệu từ JSON file"""
    print(f"\n📂 Loading products from {json_file}...")
    
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"File not found: {json_file}")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        products = json.load(f)
    
    print(f"   ✅ Loaded {len(products)} products")
    return products

def insert_products(products: List[Dict]):
    """Insert products vào database"""
    print("\n💾 Inserting products into database...")
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Prepare data for batch insert
        values = [
            (
                p['product_id'],
                p['product_name'],
                p['category'],
                p['category_id'],
                p['description'],
                p['price'],
                p['currency'],
                p['image_path'],
                p['in_stock'],
                p['rating'],
                p['reviews_count']
            )
            for p in products
        ]
        
        # Batch insert
        insert_sql = """
            INSERT INTO products (
                product_id, product_name, category, category_id,
                description, price, currency, image_path,
                in_stock, rating, reviews_count
            ) VALUES %s
            ON CONFLICT (product_id) DO UPDATE SET
                product_name = EXCLUDED.product_name,
                category = EXCLUDED.category,
                category_id = EXCLUDED.category_id,
                description = EXCLUDED.description,
                price = EXCLUDED.price,
                currency = EXCLUDED.currency,
                image_path = EXCLUDED.image_path,
                in_stock = EXCLUDED.in_stock,
                rating = EXCLUDED.rating,
                reviews_count = EXCLUDED.reviews_count,
                updated_at = CURRENT_TIMESTAMP
        """
        
        execute_values(cursor, insert_sql, values)
        conn.commit()
        
        print(f"   ✅ Inserted {len(products)} products")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()

def verify_data():
    """Kiểm tra dữ liệu đã insert"""
    print("\n🔍 Verifying data...")
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Count total products
        cursor.execute("SELECT COUNT(*) FROM products")
        total = cursor.fetchone()[0]
        print(f"   - Total products: {total}")
        
        # Count by category
        cursor.execute("""
            SELECT category, COUNT(*) as count
            FROM products
            GROUP BY category
            ORDER BY count DESC
            LIMIT 10
        """)
        
        print(f"\n   Top 10 categories:")
        for category, count in cursor.fetchall():
            print(f"      - {category}: {count}")
        
        # Price stats
        cursor.execute("""
            SELECT 
                MIN(price) as min_price,
                MAX(price) as max_price,
                AVG(price) as avg_price
            FROM products
        """)
        
        min_p, max_p, avg_p = cursor.fetchone()
        print(f"\n   Price statistics:")
        print(f"      - Min: {min_p:,} VNĐ")
        print(f"      - Max: {max_p:,} VNĐ")
        print(f"      - Avg: {avg_p:,.0f} VNĐ")
        
        # Stock status
        cursor.execute("""
            SELECT in_stock, COUNT(*) as count
            FROM products
            GROUP BY in_stock
        """)
        
        print(f"\n   Stock status:")
        for in_stock, count in cursor.fetchall():
            status = "In Stock" if in_stock else "Out of Stock"
            print(f"      - {status}: {count}")
        
    finally:
        cursor.close()
        conn.close()

def create_sample_queries():
    """Tạo một số sample queries để test"""
    print("\n📝 Sample queries:")
    
    queries = [
        ("Get product by product_id", "SELECT * FROM products WHERE product_id = 'train_00000';"),
        ("Search by category", "SELECT * FROM products WHERE category = 'apple' LIMIT 10;"),
        ("Products in stock", "SELECT * FROM products WHERE in_stock = TRUE LIMIT 10;"),
        ("Top rated products", "SELECT * FROM products ORDER BY rating DESC LIMIT 10;"),
        ("Price range", "SELECT * FROM products WHERE price BETWEEN 500000 AND 1000000 LIMIT 10;"),
    ]
    
    for name, sql in queries:
        print(f"\n   {name}:")
        print(f"   {sql}")

# ============================================================================
# Main Setup Function
# ============================================================================

def setup_database(json_file: str = 'product_metadata.json'):
    """
    Main function để setup toàn bộ database
    """
    print("=" * 70)
    print("🚀 POSTGRESQL DATABASE SETUP")
    print("=" * 70)
    
    try:
        # 1. Create database
        create_database()
        
        # 2. Create tables
        create_tables()
        
        # 3. Load data from JSON
        products = load_products_from_json(json_file)
        
        # 4. Insert products
        insert_products(products)
        
        # 5. Verify data
        verify_data()
        
        # 6. Show sample queries
        create_sample_queries()
        
        print("\n" + "=" * 70)
        print("✨ DATABASE SETUP COMPLETED!")
        print("=" * 70)
        print(f"\nConnection details:")
        print(f"  Host: {DB_CONFIG['host']}")
        print(f"  Port: {DB_CONFIG['port']}")
        print(f"  Database: {DB_CONFIG['database']}")
        print(f"  User: {DB_CONFIG['user']}")
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        raise

# ============================================================================
# Utility Functions
# ============================================================================

def reset_database():
    """Reset database (xóa tất cả dữ liệu)"""
    print("\n⚠️  WARNING: This will delete all data!")
    confirm = input("Type 'yes' to confirm: ")
    
    if confirm.lower() == 'yes':
        conn = get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("DROP TABLE IF EXISTS products CASCADE;")
            conn.commit()
            print("   ✅ Database reset completed")
        finally:
            cursor.close()
            conn.close()
    else:
        print("   ℹ️  Reset cancelled")

def export_to_json(output_file: str = 'products_export.json'):
    """Export dữ liệu từ database ra JSON"""
    print(f"\n📤 Exporting to {output_file}...")
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT * FROM products ORDER BY id")
        columns = [desc[0] for desc in cursor.description]
        
        products = []
        for row in cursor.fetchall():
            product = dict(zip(columns, row))
            # Convert datetime to string
            if 'created_at' in product:
                product['created_at'] = str(product['created_at'])
            if 'updated_at' in product:
                product['updated_at'] = str(product['updated_at'])
            products.append(product)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(products, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"   ✅ Exported {len(products)} products")
        
    finally:
        cursor.close()
        conn.close()

# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Main CLI interface"""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'setup':
            json_file = sys.argv[2] if len(sys.argv) > 2 else 'product_metadata.json'
            setup_database(json_file)
        
        elif command == 'reset':
            reset_database()
        
        elif command == 'verify':
            verify_data()
        
        elif command == 'export':
            output_file = sys.argv[2] if len(sys.argv) > 2 else 'products_export.json'
            export_to_json(output_file)
        
        else:
            print(f"Unknown command: {command}")
            print("\nAvailable commands:")
            print("  setup [json_file]  - Setup database and import data")
            print("  reset              - Reset database (delete all data)")
            print("  verify             - Verify data in database")
            print("  export [json_file] - Export database to JSON")
    
    else:
        # Default: run full setup
        setup_database()

if __name__ == "__main__":
    main()