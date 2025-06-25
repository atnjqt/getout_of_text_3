import sqlite3

def display_texts_from_db(db_name='pdf_texts.db'):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM pdf_texts')
    rows = cursor.fetchall()
    conn.close()

    if rows:
        for row in rows:
            print(f"ID: {row[0]}")
            print(f"Text: {row[1]}")
            print("-" * 40)
    else:
        print("No records found in the database.")

def main():
    try:
        display_texts_from_db()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
