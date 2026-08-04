import sqlite3
import os

DB_PATH = "/tmp/agentic_rag_demo.db"


def build_sample_database(db_path: str = DB_PATH) -> str:
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE sales (
            id INTEGER PRIMARY KEY,
            product TEXT,
            category TEXT,
            units_sold INTEGER,
            revenue REAL,
            month TEXT
        )
    """)
    sample_sales = [
        (1, "Wireless Mouse", "Electronics", 320, 4800.00, "January"),
        (2, "Mechanical Keyboard", "Electronics", 150, 7500.00, "January"),
        (3, "Desk Lamp", "Home", 210, 3150.00, "January"),
        (4, "Wireless Mouse", "Electronics", 280, 4200.00, "February"),
        (5, "Standing Desk", "Furniture", 60, 18000.00, "February"),
        (6, "Mechanical Keyboard", "Electronics", 190, 9500.00, "February"),
        (7, "Desk Lamp", "Home", 175, 2625.00, "February"),
        (8, "Standing Desk", "Furniture", 75, 22500.00, "March"),
        (9, "Wireless Mouse", "Electronics", 400, 6000.00, "March"),
        (10, "Office Chair", "Furniture", 95, 14250.00, "March"),
    ]
    cur.executemany(
        "INSERT INTO sales (id, product, category, units_sold, revenue, month) VALUES (?,?,?,?,?,?)",
        sample_sales,
    )

    cur.execute("""
        CREATE TABLE papers (
            id INTEGER PRIMARY KEY,
            title TEXT,
            field TEXT,
            year INTEGER,
            citations INTEGER
        )
    """)
    sample_papers = [
        (1, "Attention Is All You Need", "NLP", 2017, 95000),
        (2, "BERT: Pre-training of Deep Bidirectional Transformers", "NLP", 2018, 70000),
        (3, "Denoising Diffusion Probabilistic Models", "Computer Vision", 2020, 12000),
        (4, "LoRA: Low-Rank Adaptation of Large Language Models", "NLP", 2021, 6000),
        (5, "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", "NLP", 2020, 4500),
    ]
    cur.executemany(
        "INSERT INTO papers (id, title, field, year, citations) VALUES (?,?,?,?,?)",
        sample_papers,
    )

    conn.commit()
    conn.close()
    return db_path
