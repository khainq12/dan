import psycopg2

conn = psycopg2.connect(
    dbname="ai_image_db",
    user="postgres",
    password="12345",   # 🔥 sửa password của bạn
    host="localhost",
    port="5432"
)

cur = conn.cursor()

def save_to_db(path, label, confidence, risk, emb):
    try:
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
        conn.rollback()
        print("❌ DB ERROR:", e)