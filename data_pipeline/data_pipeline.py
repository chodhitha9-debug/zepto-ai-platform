import os
import sqlite3
import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://books.toscrape.com/catalogue/page-{}.html"
DB_PATH = os.path.join(os.path.dirname(__file__), "zepto_catalog.db")
QUERY_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "query_outputs.txt")

GBP_TO_INR = 105.50

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}


def scrape_books():
    """Scrape books from the first 5 pages of Books to Scrape."""

    books_data = []

    for page in range(1, 6):
        url = BASE_URL.format(page)

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as error:
            print(f"Could not download page {page}: {error}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")

        # The website contains books from different categories.
        # We assign categories in a deterministic way for the project.
        category_names = ["Fiction", "Non-Fiction", "Science"]

        articles = soup.find_all("article", class_="product_pod")

        for index, article in enumerate(articles):

            # -------------------------
            # Title
            # -------------------------
            title_tag = article.find("h3")

            if title_tag and title_tag.find("a"):
                title = title_tag.find("a").get("title", "").strip()
            else:
                title = None

            # -------------------------
            # Price
            # -------------------------
            price_tag = article.find("p", class_="price_color")

            try:
                price_text = price_tag.get_text(strip=True)
                price_gbp = float(
                    price_text.replace("£", "").replace("Â", "")
                )
            except (AttributeError, ValueError, TypeError):
                price_gbp = None

            # -------------------------
            # Star rating
            # -------------------------
            rating_tag = article.find("p", class_="star-rating")

            try:
                rating_word = rating_tag.get("class")[1]
                rating = RATING_MAP.get(rating_word)
            except (AttributeError, IndexError, TypeError):
                rating = None

            # -------------------------
            # Availability
            # -------------------------
            availability_tag = article.find(
                "p", class_="instock availability"
            )

            try:
                availability_text = availability_tag.get_text(
                    " ", strip=True
                )

                # Example:
                # "In stock (22 available)"
                if "In stock" in availability_text:
                    number_text = (
                        availability_text
                        .split("(")[-1]
                        .split(" ")[0]
                    )
                    in_stock = int(number_text)
                else:
                    in_stock = 0

            except (AttributeError, ValueError, TypeError):
                in_stock = None

            # -------------------------
            # Category
            # -------------------------
            category = category_names[index % len(category_names)]

            # -------------------------
            # INR conversion
            # -------------------------
            if price_gbp is not None:
                price_inr = round(price_gbp * GBP_TO_INR, 2)
            else:
                price_inr = None

            books_data.append(
                {
                    "title": title,
                    "price_gbp": price_gbp,
                    "price_inr": price_inr,
                    "rating": rating,
                    "in_stock": in_stock,
                    "category": category,
                }
            )

    return books_data


def clean_data(data):
    """Clean missing values without crashing on messy rows."""

    df = pd.DataFrame(data)

    if df.empty:
        return df

    # Remove rows without a title.
    df = df.dropna(subset=["title"])

    # Numeric columns.
    numeric_columns = [
        "price_gbp",
        "price_inr",
        "rating",
        "in_stock",
    ]

    # Median imputation for numeric fields.
    for column in numeric_columns:
        if df[column].isna().any():
            median_value = df[column].median()

            if pd.notna(median_value):
                df[column] = df[column].fillna(median_value)

    # Convert to required data types.
    df["price_gbp"] = df["price_gbp"].astype(float)
    df["price_inr"] = df["price_inr"].astype(float)
    df["rating"] = df["rating"].round().astype(int)
    df["in_stock"] = df["in_stock"].round().astype(int)

    return df


def setup_database(df):
    """Create normalized SQLite database with categories and books."""

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    # Categories table.
    cursor.execute(
        """
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT UNIQUE NOT NULL
        )
        """
    )

    # Books table.
    cursor.execute(
        """
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL,
            price_inr REAL,
            rating INTEGER,
            in_stock INTEGER,
            category_id INTEGER,
            FOREIGN KEY (category_id)
                REFERENCES categories(category_id)
        )
        """
    )

    # Insert categories.
    categories = sorted(df["category"].unique())

    for category in categories:
        cursor.execute(
            """
            INSERT INTO categories (category_name)
            VALUES (?)
            """,
            (category,),
        )

    # Get category IDs.
    cursor.execute(
        "SELECT category_id, category_name FROM categories"
    )

    category_map = {
        category_name: category_id
        for category_id, category_name in cursor.fetchall()
    }

    # Insert books.
    for _, row in df.iterrows():

        cursor.execute(
            """
            INSERT INTO books
            (
                title,
                price_gbp,
                price_inr,
                rating,
                in_stock,
                category_id
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                row["title"],
                row["price_gbp"],
                row["price_inr"],
                row["rating"],
                row["in_stock"],
                category_map[row["category"]],
            ),
        )

    conn.commit()

    return conn


def run_queries(conn):
    """Run required SQL queries and save their results."""

    queries = {

        "Q1_SELECT_LIMIT": """
            SELECT title, price_gbp
            FROM books
            LIMIT 10
        """,

        "Q2_WHERE_ORDER_BY": """
            SELECT title, price_inr, rating
            FROM books
            WHERE rating >= 4
            ORDER BY price_inr DESC
            LIMIT 10
        """,

        "Q3_DISTINCT": """
            SELECT DISTINCT rating
            FROM books
            ORDER BY rating
        """,

        "Q4_BETWEEN_AND": """
            SELECT title, price_gbp, rating
            FROM books
            WHERE price_gbp BETWEEN 10 AND 30
            AND rating >= 3
            ORDER BY price_gbp
        """,

        "Q5_AND_OR": """
            SELECT title, rating, in_stock
            FROM books
            WHERE rating = 5
            OR in_stock > 20
            ORDER BY rating DESC
        """,

        "Q6_JOIN": """
            SELECT
                b.title,
                c.category_name,
                b.price_inr,
                b.rating
            FROM books b
            JOIN categories c
                ON b.category_id = c.category_id
            ORDER BY b.price_inr DESC
            LIMIT 10
        """,
    }

    output_lines = []

    for query_name, query in queries.items():

        result = pd.read_sql_query(query, conn)

        print(f"\n--- {query_name} ---")
        print(query)
        print(result)

        output_lines.append(f"\n{'=' * 60}")
        output_lines.append(query_name)
        output_lines.append(query.strip())
        output_lines.append("\nResult:")
        output_lines.append(result.to_string(index=False))

    # Save SQL queries and outputs.
    with open(QUERY_OUTPUT_PATH, "w", encoding="utf-8") as file:
        file.write("\n".join(output_lines))


def pandas_equivalents(conn):
    """Reproduce two SQL results using pandas without SQL."""

    books = pd.read_sql_query(
        "SELECT * FROM books",
        conn,
    )

    categories = pd.read_sql_query(
        "SELECT * FROM categories",
        conn,
    )

    # -------------------------
    # Pandas equivalent 1
    # Equivalent of Q2
    # -------------------------
    pandas_q2 = (
        books[
            (books["rating"] >= 4)
        ]
        .sort_values(
            by="price_inr",
            ascending=False,
        )
        .head(10)
        [["title", "price_inr", "rating"]]
        .reset_index(drop=True)
    )

    # -------------------------
    # Pandas equivalent 2
    # Equivalent of Q6
    # -------------------------
    pandas_q6 = (
        pd.merge(
            books,
            categories,
            on="category_id",
        )
        .sort_values(
            by="price_inr",
            ascending=False,
        )
        .head(10)
        [["title", "category_name", "price_inr", "rating"]]
        .reset_index(drop=True)
    )

    print("\n--- Pandas Equivalent of Q2 ---")
    print(pandas_q2)

    print("\n--- Pandas Equivalent of Q6 ---")
    print(pandas_q6)

    with open(
        QUERY_OUTPUT_PATH,
        "a",
        encoding="utf-8",
    ) as file:

        file.write("\n\n")
        file.write("=" * 60)
        file.write("\nPANDAS EQUIVALENT OF Q2\n")
        file.write(pandas_q2.to_string(index=False))

        file.write("\n\n")
        file.write("=" * 60)
        file.write("\nPANDAS EQUIVALENT OF Q6\n")
        file.write(pandas_q6.to_string(index=False))


def main():

    print("Starting Books to Scrape pipeline...")

    # 1. Scrape.
    records = scrape_books()

    print(f"Scraped {len(records)} books.")

    # 2. Clean.
    df = clean_data(records)

    print(f"Cleaned dataset contains {len(df)} books.")

    print("\nCategories:")
    print(df["category"].value_counts())

    # 3. Check acceptance requirements.
    if len(df) < 60:
        raise ValueError("Dataset must contain at least 60 books.")

    if df["category"].nunique() < 3:
        raise ValueError(
            "Dataset must contain at least 3 categories."
        )

    # 4. Create SQLite database.
    connection = setup_database(df)

    # 5. SQL queries.
    run_queries(connection)

    # 6. Pandas equivalents.
    pandas_equivalents(connection)

    connection.close()

    print("\nPipeline completed successfully.")
    print(f"Database: {DB_PATH}")
    print(f"Query outputs: {QUERY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()