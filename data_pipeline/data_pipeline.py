import sqlite3
import pandas as pd
import requests
from bs4 import BeautifulSoup


def scrape_books():
    base_url = "https://books.toscrape.com/catalogue/page-{}.html"
    books_data = []
    rating_map = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

    for page in range(1, 6):
        res = requests.get(base_url.format(page))
        if res.status_code != 200:
            continue
        soup = BeautifulSoup(res.text, "html.parser")
        articles = soup.find_all("article", class_="product_pod")

        for article in articles:
            title = article.h3.a["title"]
            price_text = article.find("p", class_="price_color").text
            price_gbp = float(price_text.replace("£", "").replace("Â", ""))

            rating_class = article.find("p", class_="star-rating")["class"][1]
            rating = rating_map.get(rating_class, 3)

            avail_text = article.find(
                "p", class_="instock availability"
            ).text.strip()
            in_stock = 1 if "In stock" in avail_text else 0

            price_inr = round(price_gbp * 105.50, 2)
            category = "General"

            books_data.append(
                (title, price_gbp, price_inr, rating, in_stock, category)
            )

    return books_data


def setup_database(data):
    conn = sqlite3.connect("data_pipeline/zepto_catalog.db")
    cursor = conn.cursor()

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS categories (
        category_id INTEGER PRIMARY KEY AUTOINCREMENT,
        category_name TEXT UNIQUE
    )
    """
    )

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS books (
        book_id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        price_gbp REAL,
        price_inr REAL,
        rating INTEGER,
        in_stock INTEGER,
        category_id INTEGER,
        FOREIGN KEY (category_id) REFERENCES categories(category_id)
    )
    """
    )

    categories = list(set([row[5] for row in data]))
    for cat in categories:
        cursor.execute(
            "INSERT OR IGNORE INTO categories (category_name) VALUES (?)",
            (cat,),
        )

    cursor.execute("SELECT category_name, category_id FROM categories")
    cat_map = dict(cursor.fetchall())

    for row in data:
        title, p_gbp, p_inr, rat, stock, cat_name = row
        cat_id = cat_map[cat_name]
        cursor.execute(
            """
        INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
            (title, p_gbp, p_inr, rat, stock, cat_id),
        )

    conn.commit()
    return conn


def run_queries(conn):
    q1 = """
    SELECT b.title, c.category_name, b.price_inr, b.rating 
    FROM books b 
    JOIN categories c ON b.category_id = c.category_id 
    ORDER BY b.rating DESC LIMIT 5
    """

    df_sql = pd.read_sql(q1, conn)

    df_books = pd.read_sql("SELECT * FROM books", conn)
    df_cats = pd.read_sql("SELECT * FROM categories", conn)
    df_merged = pd.merge(df_books, df_cats, on="category_id")
    df_merged = (
        df_merged.sort_values(by="rating", ascending=False)
        .head(5)[["title", "category_name", "price_inr", "rating"]]
        .reset_index(drop=True)
    )

    print("--- SQL Query Result ---")
    print(df_sql)
    print("\n--- Pandas Merge Verification ---")
    print(df_merged)


if __name__ == "__main__":
    records = scrape_books()
    connection = setup_database(records)
    run_queries(connection)
    connection.close()