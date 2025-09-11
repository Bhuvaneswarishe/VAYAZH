import sqlite3

def create_tables():
    conn = sqlite3.connect("farmers.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS farmer_details (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        location TEXT,
        landSize TEXT,
        soilType TEXT,
        irrigationMethod TEXT,
        waterSource TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        farmer_id INTEGER,
        question TEXT,
        answer TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(farmer_id) REFERENCES farmer_details(id)
    )
    """)

    conn.commit()
    conn.close()

    
from database import create_tables
create_tables()

