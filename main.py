from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import torch
import clip
import faiss
import numpy as np
import pickle
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
from PIL import Image
import io
import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from datetime import datetime

# ============================================================================
# Load environments variables
# ============================================================================
load_dotenv()

# ============================================================================
# Pydantic Models
# ============================================================================

class ProductResponse(BaseModel):
    """Model cho thông tin sản phẩm"""
    id: int
    product_id: str
    product_name: str
    category: str
    category_id: int
    description: str
    price: int
    currency: str
    image_path: str
    in_stock: bool
    rating: float
    reviews_count: int
    similarity_score: Optional[float] = None
    distance: Optional[float] = None

class SearchResponse(BaseModel):
    """Model cho kết quả tìm kiếm"""
    query_type: str
    total_results: int
    search_time_ms: float
    products: List[ProductResponse]

class HealthResponse(BaseModel):
    """Model cho health check"""
    status: str
    models_loaded: bool
    index_loaded: bool
    database_connected: bool
    total_products: int
    feature_dimension: int

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
# Global Variables & Configuration
# ============================================================================

class AppState:
    """Lưu trữ state của ứng dụng"""
    clip_model = None
    clip_preprocess = None
    faiss_index = None
    product_ids: List[str] = []
    id_to_index: Dict[str, int] = {}
    feature_dim: int = 0
    device: str = "cpu"
    db_pool: SimpleConnectionPool = None

state = AppState()

CONFIG = {
    "clip_model": os.getenv('CLIP_MODEL', 'ViT-B/32'),
    "faiss_index_path": os.getenv('FAISS_INDEX_PATH', './faiss_indexes/index_hnsw_flat.index'),
    "features_dir": os.getenv('FEATURES_DIR', './features'),
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "db_pool_min": int(os.getenv('DB_POOL_MIN', 2)),
    "db_pool_max": int(os.getenv('DB_POOL_MAX', 10))
}

# ============================================================================
# Database Functions
# ============================================================================

def get_db_connection():
    """Lấy connection từ pool"""
    return state.db_pool.getconn()

def release_db_connection(conn):
    """Trả connection về pool"""
    state.db_pool.putconn(conn)

def get_db():
    """Dependency để lấy database connection"""
    conn = get_db_connection()
    try:
        conn.autocommit = True 
        yield conn
    finally:
        release_db_connection(conn)

def init_db_pool():
    """Khởi tạo connection pool"""
    print(f"\n💾 Initializing database connection pool...")
    
    try:
        state.db_pool = SimpleConnectionPool(
            CONFIG['db_pool_min'],
            CONFIG['db_pool_max'],
            **DB_CONFIG
        )
        
        # Test connection
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM products")
        count = cursor.fetchone()[0]
        cursor.close()
        release_db_connection(conn)
        
        print(f"   ✅ Database connected")
        print(f"   - Total products in DB: {count}")
        
    except Exception as e:
        print(f"   ❌ Database connection failed: {e}")
        raise

def close_db_pool():
    """Đóng connection pool"""
    if state.db_pool:
        state.db_pool.closeall()
        print("   ✅ Database pool closed")

# ============================================================================
# Lifespan Events
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management"""
    print("=" * 70)
    print("🚀 STARTING IMAGE RETRIEVAL API WITH POSTGRESQL")
    print("=" * 70)
    
    try:
        await load_clip_model()
        await load_faiss_index()
        init_db_pool()
        
        print("\n✅ All systems ready!")
        print(f"   - CLIP Model: {CONFIG['clip_model']}")
        print(f"   - FAISS vectors: {len(state.product_ids)}")
        print(f"   - Feature dimension: {state.feature_dim}")
        print(f"   - Device: {state.device}")
        print(f"   - Database: {DB_CONFIG['database']}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Startup error: {e}")
        raise
    
    yield
    
    print("\n🛑 Shutting down...")
    state.clip_model = None
    state.faiss_index = None
    close_db_pool()
    print("✅ Cleanup completed")

# ============================================================================
# Initialization Functions
# ============================================================================

async def load_clip_model():
    """Load CLIP model"""
    print(f"\n📦 Loading CLIP model: {CONFIG['clip_model']}...")
    
    state.device = CONFIG['device']
    state.clip_model, state.clip_preprocess = clip.load(
        CONFIG['clip_model'], 
        device=state.device
    )
    state.clip_model.eval()
    
    print(f"   ✅ CLIP model loaded on {state.device}")

async def load_faiss_index():
    """Load FAISS index"""
    print(f"\n📦 Loading FAISS index: {CONFIG['faiss_index_path']}...")
    
    if not os.path.exists(CONFIG['faiss_index_path']):
        raise FileNotFoundError(f"FAISS index not found: {CONFIG['faiss_index_path']}")
    
    state.faiss_index = faiss.read_index(CONFIG['faiss_index_path'])
    
    # Load product IDs
    ids_file = os.path.join(CONFIG['features_dir'], 'product_ids.pkl')
    with open(ids_file, 'rb') as f:
        state.product_ids = pickle.load(f)
    
    # Load id to index mapping
    index_file = os.path.join(CONFIG['features_dir'], 'id_to_index.pkl')
    with open(index_file, 'rb') as f:
        state.id_to_index = pickle.load(f)
    
    # Load features metadata
    import json
    meta_file = os.path.join(CONFIG['features_dir'], 'features_metadata.json')
    with open(meta_file, 'r') as f:
        meta = json.load(f)
        state.feature_dim = meta['feature_dim']
    
    print(f"   ✅ FAISS index loaded")
    print(f"   - Total vectors: {state.faiss_index.ntotal}")
    print(f"   - Feature dimension: {state.feature_dim}")

# ============================================================================
# Helper Functions
# ============================================================================

def extract_image_features(image: Image.Image) -> np.ndarray:
    """Trích xuất features từ ảnh bằng CLIP"""
    image_input = state.clip_preprocess(image).unsqueeze(0).to(state.device)
    
    with torch.no_grad():
        features = state.clip_model.encode_image(image_input)
        features = features / features.norm(dim=-1, keepdim=True)
    
    return features.cpu().numpy().flatten()

def extract_text_features(text: str) -> np.ndarray:
    """Trích xuất features từ text bằng CLIP"""
    text_input = clip.tokenize([text]).to(state.device)
    
    with torch.no_grad():
        features = state.clip_model.encode_text(text_input)
        features = features / features.norm(dim=-1, keepdim=True)
    
    return features.cpu().numpy().flatten()

def search_faiss(query_vector: np.ndarray, k: int = 10) -> tuple:
    """Tìm kiếm trong FAISS index"""
    query_vector = query_vector.reshape(1, -1).astype(np.float32)
    distances, indices = state.faiss_index.search(query_vector, k)
    return distances[0], indices[0]

def get_products_by_product_ids(conn, product_ids: List[str]) -> List[Dict]:
    """Lấy thông tin sản phẩm từ database theo product_ids"""
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Sử dụng ANY để query với list
        cursor.execute(
            "SELECT * FROM products WHERE product_id = ANY(%s)",
            (product_ids,)
        )
        
        products = cursor.fetchall()
        
        # Tạo dict để map nhanh
        product_dict = {p['product_id']: dict(p) for p in products}
        
        # Trả về theo thứ tự của product_ids
        return [product_dict.get(pid) for pid in product_ids if pid in product_dict]
        
    finally:
        cursor.close()

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Image Retrieval API with PostgreSQL",
    description="API tìm kiếm sản phẩm bằng ảnh hoặc text - CLIP + FAISS + PostgreSQL",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Static Files Mounting
# ============================================================================

# Kiểm tra thư mục images có tồn tại không, nếu không thì tạo mới để tránh lỗi
if not os.path.exists("images"):
    os.makedirs("images")

# Mount thư mục images vào đường dẫn /images
app.mount("/images", StaticFiles(directory="images"), name="images")

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["General"])
def root():
    """Root endpoint"""
    return {
        "message": "Image Retrieval API with PostgreSQL",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    """Health check endpoint"""
    db_connected = False
    total_products = 0
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM products")
        total_products = cursor.fetchone()[0]
        cursor.close()
        release_db_connection(conn)
        db_connected = True
    except:
        pass
    
    return HealthResponse(
        status="healthy" if (state.clip_model and state.faiss_index and db_connected) else "unhealthy",
        models_loaded=state.clip_model is not None,
        index_loaded=state.faiss_index is not None,
        database_connected=db_connected,
        total_products=total_products,
        feature_dimension=state.feature_dim
    )

@app.post("/search/image", response_model=SearchResponse, tags=["Search"])
def search_by_image(
    file: UploadFile = File(..., description="Upload ảnh để tìm kiếm"),
    k: int = Query(10, ge=1, le=100, description="Số kết quả trả về"),
    conn = Depends(get_db)
):
    """Tìm kiếm sản phẩm bằng ảnh"""
    import time
    start_time = time.time()
    
    try:
        # Đọc và xử lý ảnh
        contents = file.file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Trích xuất features
        query_vector = extract_image_features(image)
        
        # Tìm kiếm trong FAISS
        distances, indices = search_faiss(query_vector, k=k)
        
        # Lấy product_ids từ indices
        product_ids = [state.product_ids[int(idx)] for idx in indices if int(idx) < len(state.product_ids)]
        
        # Query database
        products = get_products_by_product_ids(conn, product_ids)
        
        # Thêm similarity scores
        product_dict = {p['product_id']: p for p in products}
        results = []
        
        for dist, pid in zip(distances, product_ids):
            if pid in product_dict:
                product = product_dict[pid]
                similarity = 1 / (1 + float(dist))
                
                results.append(ProductResponse(
                    **product,
                    similarity_score=similarity,
                    distance=float(dist)
                ))
        
        search_time = (time.time() - start_time) * 1000
        
        return SearchResponse(
            query_type="image",
            total_results=len(results),
            search_time_ms=search_time,
            products=results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/search/text", response_model=SearchResponse, tags=["Search"])
def search_by_text(
    query: str = Query(..., description="Text query để tìm kiếm", min_length=1),
    k: int = Query(10, ge=1, le=100, description="Số kết quả trả về"),
    conn = Depends(get_db)
):
    """Tìm kiếm sản phẩm bằng text"""
    import time
    start_time = time.time()
    
    try:
        # Trích xuất text features
        query_vector = extract_text_features(query)
        
        # Tìm kiếm trong FAISS
        distances, indices = search_faiss(query_vector, k=k)
        
        # Lấy product_ids
        product_ids = [state.product_ids[int(idx)] for idx in indices if int(idx) < len(state.product_ids)]
        
        # Query database
        products = get_products_by_product_ids(conn, product_ids)
        
        # Thêm similarity scores
        product_dict = {p['product_id']: p for p in products}
        results = []
        
        for dist, pid in zip(distances, product_ids):
            if pid in product_dict:
                product = product_dict[pid]
                similarity = 1 / (1 + float(dist))
                
                results.append(ProductResponse(
                    **product,
                    similarity_score=similarity,
                    distance=float(dist)
                ))
        
        search_time = (time.time() - start_time) * 1000
        
        return SearchResponse(
            query_type="text",
            total_results=len(results),
            search_time_ms=search_time,
            products=results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")

@app.get("/products/{product_id}", response_model=ProductResponse, tags=["Products"])
def get_product(product_id: str, conn = Depends(get_db)):
    """Lấy chi tiết sản phẩm theo product_id"""
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        cursor.execute("SELECT * FROM products WHERE product_id = %s", (product_id,))
        product = cursor.fetchone()
        
        if not product:
            raise HTTPException(status_code=404, detail=f"Product not found: {product_id}")
        
        return ProductResponse(**dict(product))
        
    finally:
        cursor.close()

@app.get("/products", response_model=List[ProductResponse], tags=["Products"])
def list_products(
    skip: int = Query(0, ge=0, description="Số sản phẩm bỏ qua"),
    limit: int = Query(20, ge=1, le=100, description="Số sản phẩm trả về"),
    category: Optional[str] = Query(None, description="Lọc theo category"),
    in_stock: Optional[bool] = Query(None, description="Lọc theo trạng thái kho"),
    min_price: Optional[int] = Query(None, description="Giá tối thiểu"),
    max_price: Optional[int] = Query(None, description="Giá tối đa"),
    conn = Depends(get_db)
):
    """Liệt kê danh sách sản phẩm với filters"""
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Build query
        conditions = []
        params = []
        
        if category:
            conditions.append("category = %s")
            params.append(category)
        
        if in_stock is not None:
            conditions.append("in_stock = %s")
            params.append(in_stock)
        
        if min_price is not None:
            conditions.append("price >= %s")
            params.append(min_price)
        
        if max_price is not None:
            conditions.append("price <= %s")
            params.append(max_price)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f"""
            SELECT * FROM products 
            WHERE {where_clause}
            ORDER BY id
            LIMIT %s OFFSET %s
        """
        
        params.extend([limit, skip])
        
        cursor.execute(query, params)
        products = cursor.fetchall()
        
        return [ProductResponse(**dict(p)) for p in products]
        
    finally:
        cursor.close()

@app.get("/categories", tags=["Products"])
def get_categories(conn = Depends(get_db)):
    """Lấy danh sách tất cả categories"""
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        cursor.execute("""
            SELECT category, COUNT(*) as count
            FROM products
            GROUP BY category
            ORDER BY count DESC
        """)
        
        categories = cursor.fetchall()
        
        return {
            "total_categories": len(categories),
            "categories": [
                {"name": cat["category"], "count": cat["count"]}
                for cat in categories
            ]
        }
        
    finally:
        cursor.close()

@app.get("/stats", tags=["General"])
def get_stats(conn = Depends(get_db)):
    """Lấy thống kê tổng quan của hệ thống"""
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Total products
        cursor.execute("SELECT COUNT(*) as total FROM products")
        total = cursor.fetchone()["total"]
        
        # Categories
        cursor.execute("SELECT COUNT(DISTINCT category) as total FROM products")
        total_categories = cursor.fetchone()["total"]
        
        # Stock status
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN in_stock THEN 1 ELSE 0 END) as in_stock,
                SUM(CASE WHEN NOT in_stock THEN 1 ELSE 0 END) as out_of_stock
            FROM products
        """)
        stock = cursor.fetchone()
        
        # Price stats
        cursor.execute("""
            SELECT MIN(price) as min, MAX(price) as max, AVG(price) as avg
            FROM products
        """)
        price_stats = cursor.fetchone()
        
        # Top categories
        cursor.execute("""
            SELECT category, COUNT(*) as count
            FROM products
            GROUP BY category
            ORDER BY count DESC
            LIMIT 10
        """)
        top_categories = cursor.fetchall()
        
        return {
            "total_products": total,
            "total_categories": total_categories,
            "in_stock": stock["in_stock"],
            "out_of_stock": stock["out_of_stock"],
            "price_stats": {
                "min": price_stats["min"],
                "max": price_stats["max"],
                "avg": float(price_stats["avg"])
            },
            "top_categories": [
                {"name": cat["category"], "count": cat["count"]}
                for cat in top_categories
            ]
        }
        
    finally:
        cursor.close()

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 70)
    print("🚀 Starting FastAPI Server with PostgreSQL")
    print("=" * 70)
    print(f"📍 API URL: http://localhost:8000")
    print(f"📚 Docs: http://localhost:8000/docs")
    print(f"🔍 ReDoc: http://localhost:8000/redoc")
    print("=" * 70 + "\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )