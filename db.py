import psycopg2
from psycopg2 import pool

# ===== Connection Pool =====
_pool = None

def get_pool():
    """Khởi tạo connection pool lazily (chỉ khi cần)."""
    global _pool
    if _pool is None:
        _pool = pool.SimpleConnectionPool(
            minconn=1,
            maxconn=5,
            dbname="ai_image_db",
            user="postgres",
            password="12345",
            host="localhost",
            port="5432"
        )
    return _pool


def db_available():
    """Kiểm tra DB có kết nối được không."""
    try:
        p = get_pool()
        conn = p.getconn()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        p.putconn(conn)
        return True
    except Exception:
        return False


# =========================================================
# SCHEMA MIGRATION
# =========================================================
def _ensure_column(cur, table, column, col_type):
    """Thêm column nếu chưa tồn tại."""
    cur.execute(
        "SELECT 1 FROM information_schema.columns WHERE table_name=%s AND column_name=%s",
        (table, column)
    )
    if not cur.fetchone():
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        print(f"  + Added column {table}.{column}")


def init_db():
    """Tạo bảng nếu chưa có, thêm column nếu thiếu. An toàn cho data cũ."""
    try:
        p = get_pool()
        conn = p.getconn()
        cur = conn.cursor()

        # Tạo bảng (nếu chưa có)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id SERIAL PRIMARY KEY,
                path TEXT NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                image_id INTEGER REFERENCES images(id),
                label TEXT NOT NULL,
                confidence FLOAT NOT NULL,
                risk TEXT NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id SERIAL PRIMARY KEY,
                image_id INTEGER REFERENCES images(id),
                vector FLOAT[] NOT NULL
            )
        """)

        # Thêm column mới vào bảng cũ
        _ensure_column(cur, 'images', 'image_bytes', 'BYTEA')
        _ensure_column(cur, 'images', 'filename', 'TEXT')
        _ensure_column(cur, 'images', 'created_at', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP')
        _ensure_column(cur, 'predictions', 'created_at', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP')

        conn.commit()
        print("✅ DB schema ready")

    except Exception as e:
        print("❌ DB init error:", e)
    finally:
        if 'cur' in locals() and cur:
            cur.close()
        if 'conn' in locals() and conn:
            p.putconn(conn)


# =========================================================
# SAVE (ghi kết quả dự đoán)
# =========================================================
def save_to_db(path, label, confidence, risk, emb, image_bytes=None, filename=None):
    """Lưu kết quả dự đoán + embedding + ảnh (tuỳ chọn)."""
    try:
        p = get_pool()
        conn = p.getconn()
        cur = conn.cursor()

        # DEDUP
        cur.execute(
            "SELECT id FROM images WHERE path = %s LIMIT 1",
            (path,)
        )
        existing = cur.fetchone()
        if existing:
            print(f"ℹ️ Image already in DB (image_id={existing[0]}), skipping insert.")
            return existing[0]

        # Insert image
        cur.execute(
            "INSERT INTO images(path, image_bytes, filename) VALUES (%s, %s, %s) RETURNING id",
            (path, image_bytes, filename)
        )
        image_id = cur.fetchone()[0]

        # Insert prediction
        cur.execute(
            "INSERT INTO predictions(image_id, label, confidence, risk) VALUES (%s, %s, %s, %s)",
            (image_id, label, confidence, risk)
        )

        # Insert embedding
        cur.execute(
            "INSERT INTO embeddings(image_id, vector) VALUES (%s, %s)",
            (image_id, emb.tolist())
        )

        conn.commit()
        print(f"✅ Saved to DB (image_id={image_id})")
        return image_id

    except Exception as e:
        if 'conn' in locals():
            conn.rollback()
        print("❌ DB ERROR:", e)
    finally:
        if 'cur' in locals() and cur:
            cur.close()
        if 'conn' in locals() and conn:
            p.putconn(conn)


# =========================================================
# QUERY: LỊCH SỬ KIỂM TRA
# =========================================================
def get_history(limit=50, offset=0):
    """Lấy lịch sử dự đoán, mới nhất trước."""
    try:
        p = get_pool()
        conn = p.getconn()
        cur = conn.cursor()

        cur.execute("""
            SELECT i.id, i.path, i.image_bytes, i.filename,
                   COALESCE(i.created_at::text, 'N/A'),
                   p.label, p.confidence, p.risk
            FROM images i
            JOIN predictions p ON p.image_id = i.id
            ORDER BY COALESCE(i.created_at, NOW()) DESC
            LIMIT %s OFFSET %s
        """, (limit, offset))

        rows = cur.fetchall()
        cur.close()
        p.putconn(conn)

        return [{
            'id': r[0],
            'path': r[1],
            'image_bytes': r[2],
            'filename': r[3],
            'created_at': r[4],
            'label': r[5],
            'confidence': float(r[6]),
            'risk': r[7],
        } for r in rows]
    except Exception as e:
        print("❌ get_history error:", e)
        return []


def get_total_count():
    """Tổng số ảnh đã kiểm tra."""
    try:
        p = get_pool()
        conn = p.getconn()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM images")
        count = cur.fetchone()[0]
        cur.close()
        p.putconn(conn)
        return count
    except Exception:
        return 0


# =========================================================
# QUERY: THỐNG KÊ
# =========================================================
def get_stats():
    """Thống kê tổng quan: real/fake counts, avg confidence, risk distribution."""
    try:
        p = get_pool()
        conn = p.getconn()
        cur = conn.cursor()

        # Tổng theo label
        cur.execute("""
            SELECT label, COUNT(*), ROUND(AVG(confidence)::numeric, 4)
            FROM predictions
            GROUP BY label
        """)
        label_stats = {r[0]: {'count': r[1], 'avg_conf': float(r[2])} for r in cur.fetchall()}

        # Tổng theo risk
        cur.execute("""
            SELECT risk, COUNT(*)
            FROM predictions
            GROUP BY risk
        """)
        risk_stats = {r[0]: r[1] for r in cur.fetchall()}

        # Confidence min/max/avg toàn bộ
        cur.execute("""
            SELECT ROUND(MIN(confidence)::numeric, 4),
                   ROUND(MAX(confidence)::numeric, 4),
                   ROUND(AVG(confidence)::numeric, 4)
            FROM predictions
        """)
        row = cur.fetchone()
        conf_stats = {
            'min': float(row[0]) if row[0] else 0,
            'max': float(row[1]) if row[1] else 0,
            'avg': float(row[2]) if row[2] else 0,
        }

        cur.close()
        p.putconn(conn)

        return {
            'label_stats': label_stats,
            'risk_stats': risk_stats,
            'conf_stats': conf_stats,
        }
    except Exception as e:
        print("❌ get_stats error:", e)
        return {'label_stats': {}, 'risk_stats': {}, 'conf_stats': {}}


def get_confidence_data():
    """Lấy danh sách confidence theo label (cho histogram)."""
    try:
        p = get_pool()
        conn = p.getconn()
        cur = conn.cursor()

        cur.execute("""
            SELECT label, confidence
            FROM predictions
            ORDER BY confidence
        """)

        data = {'fake': [], 'real': []}
        for label, conf in cur.fetchall():
            if label in data:
                data[label].append(float(conf))

        cur.close()
        p.putconn(conn)
        return data
    except Exception as e:
        print("❌ get_confidence_data error:", e)
        return {'fake': [], 'real': []}


def get_daily_counts(days=30):
    """Số lượng dự đoán theo ngày (cho timeline chart)."""
    try:
        p = get_pool()
        conn = p.getconn()
        cur = conn.cursor()

        cur.execute("""
            SELECT DATE(i.created_at) AS day,
                   COUNT(*) AS total,
                   COUNT(*) FILTER (WHERE p.label = 'fake') AS fake_count,
                   COUNT(*) FILTER (WHERE p.label = 'real') AS real_count
            FROM images i
            JOIN predictions p ON p.image_id = i.id
            WHERE i.created_at IS NOT NULL
              AND i.created_at >= CURRENT_DATE - INTERVAL '%s days'
            GROUP BY day
            ORDER BY day
        """, (days,))

        rows = cur.fetchall()
        cur.close()
        p.putconn(conn)

        return [{
            'date': str(r[0]),
            'total': r[1],
            'fake': r[2],
            'real': r[3],
        } for r in rows]
    except Exception as e:
        print("❌ get_daily_counts error:", e)
        return []


# =========================================================
# QUERY: EMBEDDING SEARCH (tìm ảnh tương tự)
# =========================================================
def search_similar(query_emb, k=10):
    """Tìm ảnh tương tự trong DB dựa trên cosine similarity."""
    try:
        p = get_pool()
        conn = p.getconn()
        cur = conn.cursor()

        cur.execute("""
            SELECT i.id, i.path, i.image_bytes, i.filename,
                   p.label, p.confidence, e.vector
            FROM embeddings e
            JOIN images i ON i.id = e.image_id
            JOIN predictions p ON p.image_id = i.id
        """)

        rows = cur.fetchall()
        cur.close()
        p.putconn(conn)

        if not rows:
            return []

        query = np.array(query_emb, dtype="float32")
        results = []

        for row in rows:
            db_emb = np.array(row[6], dtype="float32")
            sim = float(np.dot(query, db_emb) / (
                np.linalg.norm(query) * np.linalg.norm(db_emb) + 1e-8
            ))
            results.append({
                'id': row[0],
                'path': row[1],
                'image_bytes': row[2],
                'filename': row[3],
                'label': row[4],
                'confidence': float(row[5]),
                'similarity': sim,
            })

        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:k]

    except Exception as e:
        print("❌ search_similar error:", e)
        return []


# =========================================================
# HELPER: hiển thị ảnh từ DB
# =========================================================
def get_image_display(record):
    """Trả về đường dẫn file hoặc bytes để hiển thị ảnh.
    Ưu tiên file path > image_bytes trong DB.
    """
    # Thử đọc từ file path trước
    if record.get('path') and os.path.exists(record['path']):
        return record['path']
    # Fallback: image bytes từ DB
    if record.get('image_bytes'):
        return record['image_bytes']
    return None


# cần import os cho get_image_display
import os
