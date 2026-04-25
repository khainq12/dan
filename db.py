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
            password="12345",   # TODO: đổi thành env var
            host="localhost",
            port="5432"
        )
    return _pool

def save_to_db(path, label, confidence, risk, emb):
    try:
        p = get_pool()
        conn = p.getconn()
        cur = conn.cursor()

        # ===== 1. insert image =====
        cur.execute(
            "INSERT INTO images(path) VALUES (%s) RETURNING id",
            (path,)
        )
        image_id = cur.fetchone()[0]

        # ===== 2. insert prediction =====
        cur.execute(
            """
            INSERT INTO predictions(image_id, label, confidence, risk)
            VALUES (%s, %s, %s, %s)
            """,
            (image_id, label, confidence, risk)
        )

        # ===== 3. insert embedding =====
        cur.execute(
            """
            INSERT INTO embeddings(image_id, vector)
            VALUES (%s, %s)
            """,
            (image_id, emb.tolist())
        )

        conn.commit()
        print(f"✅ Saved to DB (image_id={image_id})")

    except Exception as e:
        if 'conn' in locals():
            conn.rollback()
        print("❌ DB ERROR:", e)
    finally:
        if 'cur' in locals() and cur:
            cur.close()
        if 'conn' in locals() and conn:
            p.putconn(conn)
