from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "memory-db"
IRIS_CODES_DIR = DB_DIR / "iris-codes"
DATABASE_MAP_JSON = DB_DIR / "database_map.json"

if __name__ == "__main__":
    print(f"Projekt Root:  {BASE_DIR}")
    print(f"Data:          {DATA_DIR}")
    print(f"Database file: {DATABASE_MAP_JSON}")