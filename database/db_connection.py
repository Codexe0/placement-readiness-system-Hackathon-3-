import sqlite3
import pandas as pd

conn = sqlite3.connect("readiness.db")
df = pd.read_csv("data/readiness_data.csv")
df.to_sql("students_readiness", conn, if_exists="replace", index=False)
conn.close()
print("Data stored in SQLite.")
