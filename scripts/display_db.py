import sqlite3
from bs4 import BeautifulSoup

def display_texts_from_db(db_name='pdf_texts.db'):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # Check if the updated schema is in place
    cursor.execute("PRAGMA table_info(pdf_texts)")
    columns = [column[1] for column in cursor.fetchall()]
    
    cursor.execute('SELECT * FROM pdf_texts')
    rows = cursor.fetchall()
    conn.close()

    if rows:
        for row in rows:
            print(f"ID: {row[0]}")
            
            # Display text (always present)
            print(f"Text: {row[1][:100]}..." if len(row[1]) > 100 else f"Text: {row[1]}")
            
            # Display URL and timestamp if they exist in the schema
            if 'url' in columns and len(row) > 2:
                print(f"URL: {row[2]}")
            if 'timestamp' in columns and len(row) > 3:
                print(f"Downloaded at: {row[3]}")
            
            # Display HTML metadata if it exists
            if 'html_metadata' in columns and len(row) > 4 and row[4]:
                print("HTML Metadata:")
                # Try to parse the HTML and extract key info
                try:
                    soup = BeautifulSoup(row[4], 'html.parser')
                    # Extract metadata in a more readable format
                    for div in soup.find_all('div', class_='flex-col'):
                        label = div.find('strong')
                        value = div.find('span')
                        if label and value:
                            print(f"  {label.text.strip()}: {value.text.strip()}")
                except Exception:
                    # If parsing fails, just show the first part of the raw HTML
                    print(f"  Raw metadata: {row[4][:100]}..." if len(row[4]) > 100 else f"  Raw metadata: {row[4]}")
                
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
