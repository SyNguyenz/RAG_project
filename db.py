import json
import psycopg2
import os

# ================= CẤU HÌNH =================
# Copy "External Database URL" từ Render Dashboard
# Ví dụ: postgres://user:password@host.render.com/db_name
DB_URL = "postgresql://synz:9VzDdpjttUbv6mPI9lxPnHbWpv07xjua@dpg-d4qi52m3jp1c739hq5dg-a.virginia-postgres.render.com/tvdl"

# Tên file JSON chứa dữ liệu
JSON_FILE = "product_metadata.json" 

def import_data():
    try:
        print("🔌 Đang kết nối đến Database...")
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        print("✅ Kết nối thành công!")

        # --- BƯỚC MỚI: KIỂM TRA TIẾN ĐỘ ---
        print("🔍 Đang kiểm tra dữ liệu hiện có...")
        cur.execute("SELECT MAX(id) FROM products;")
        row = cur.fetchone()
        
        # Nếu chưa có gì thì start = 0, nếu có rồi thì start = max_id
        current_max_id = row[0] if row[0] is not None else 0
        
        print(f"ℹ️ Database đang dừng ở ID: {current_max_id}")

        # 2. Đọc file JSON
        if not os.path.exists(JSON_FILE):
            print(f"❌ Không tìm thấy file {JSON_FILE}")
            return

        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            products = json.load(f)
        
        # Chỉ lấy những phần tử chưa được insert
        # Ví dụ: Database có 500 dòng (id 1->500). List products sẽ cắt từ index 500 trở đi.
        remaining_products = products[current_max_id:] 
        
        if not remaining_products:
            print("✅ Dữ liệu đã đầy đủ. Không cần import thêm!")
            return

        print(f"📦 Tổng file JSON: {len(products)}. Cần import tiếp: {len(remaining_products)} dòng.")

        # 3. Duyệt và Insert phần còn lại
        count = 0
        
        # Lưu ý: enumerate bắt đầu đếm từ con số current_max_id để ID luôn đúng
        for i, item in enumerate(remaining_products, start=current_max_id):
            
            # Xử lý đường dẫn
            clean_path = item.get('image_path', '').replace('./', '')

            sql = """
                INSERT INTO products (
                    id, product_id, image_path, product_name, category, category_id, 
                    description, price, currency, in_stock, rating, reviews_count
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """
            
            # i chạy từ 500 -> id sẽ là 501 (đúng logic +1)
            new_id = i + 1 

            values = (
                new_id,
                item.get('product_id'),
                clean_path,
                item.get('product_name'),
                item.get('category'),
                item.get('category_id'),
                item.get('description'),
                item.get('price'),
                item.get('currency'),
                item.get('in_stock'),
                item.get('rating'),
                item.get('reviews_count')
            )

            cur.execute(sql, values)
            count += 1
            
            if count % 100 == 0:
                print(f"   Writing ID {new_id}... ({count}/{len(remaining_products)})")

        conn.commit()
        print(f"🎉 Đã import xong {count} dòng mới!")

    except Exception as e:
        print("❌ Có lỗi xảy ra:", e)
        if conn:
            conn.rollback()
    finally:
        if conn:
            cur.close()
            conn.close()

if __name__ == "__main__":
    import_data()