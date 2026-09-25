import sqlite3
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://books.toscrape.com/catalogue/page-{}.html"
GBP_TO_INR = 105.50

DB_PATH = "data_pipeline/books.db"
RAW_CSV_PATH = "data_pipeline/books_raw.csv"
CLEAN_CSV_PATH = "data_pipeline/books_clean.csv"
SQL_RESULTS_PATH = "data_pipeline/sql_results.txt"

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}


def create_session():
    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/154.0.0.0 Safari/537.36"
            )
        }
    )

    return session


def scrape_book_details(session, detail_url):
    response = session.get(
        detail_url,
        timeout=20
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.content,
        "html.parser"
    )

    title_tag = soup.find("h1")

    price_tag = soup.find(
        "p",
        class_="price_color"
    )

    availability_tag = soup.find(
        "p",
        class_="instock availability"
    )

    rating_tag = soup.find(
        "p",
        class_="star-rating"
    )

    if title_tag is None:
        raise ValueError(
            "Book title could not be found."
        )

    if price_tag is None:
        raise ValueError(
            "Book price could not be found."
        )

    if availability_tag is None:
        raise ValueError(
            "Book availability could not be found."
        )

    if rating_tag is None:
        raise ValueError(
            "Book rating could not be found."
        )

    title = title_tag.get_text(
        strip=True
    )

    price = price_tag.get_text(
        strip=True
    )

    availability = availability_tag.get_text(
        " ",
        strip=True
    )

    rating_classes = rating_tag.get(
        "class",
        []
    )

    rating_word = next(
        (
            value
            for value in rating_classes
            if value in RATING_MAP
        ),
        None,
    )

    if rating_word is None:
        raise ValueError(
            f"Unknown rating for book: {title}"
        )

    category = "Unknown"

    breadcrumb = soup.find(
        "ul",
        class_="breadcrumb"
    )

    if breadcrumb is not None:
        links = breadcrumb.find_all("a")

        if len(links) >= 2:
            category = links[-1].get_text(
                strip=True
            )

    return {
        "title": title,
        "price": price,
        "star_rating": rating_word,
        "availability": availability,
        "category": category,
        "url": detail_url,
    }


def scrape_books():
    session = create_session()
    books = []

    for page_number in range(1, 6):
        url = BASE_URL.format(page_number)

        response = session.get(
            url,
            timeout=20
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.content,
            "html.parser"
        )

        articles = soup.find_all(
            "article",
            class_="product_pod"
        )

        page_count = 0

        for article in articles:
            heading = article.find("h3")

            if heading is None:
                continue

            link = heading.find("a")

            if link is None:
                continue

            detail_link = link.get("href")

            if not detail_link:
                continue

            detail_url = urljoin(
                url,
                detail_link
            )

            try:
                book = scrape_book_details(
                    session,
                    detail_url
                )

                books.append(book)
                page_count += 1

            except Exception as error:
                print(
                    f"Skipping book: {error}"
                )

        print(
            f"Page {page_number}: "
            f"{page_count} books scraped"
        )

    return pd.DataFrame(books)


def clean_price(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    value = (
        value
        .replace("Â£", "")
        .replace("£", "")
        .replace("&pound;", "")
        .replace("GBP", "")
        .replace(",", "")
        .strip()
    )

    try:
        return float(value)

    except ValueError:
        return None


def clean_data(df):
    clean_df = df.copy()

    clean_df["price_gbp"] = (
        clean_df["price"]
        .apply(clean_price)
    )

    price_median = clean_df[
        "price_gbp"
    ].median()

    clean_df["price_gbp"] = (
        clean_df["price_gbp"]
        .fillna(price_median)
    )

    clean_df["rating"] = (
        clean_df["star_rating"]
        .map(RATING_MAP)
    )

    rating_mode = clean_df[
        "rating"
    ].mode()[0]

    clean_df["rating"] = (
        clean_df["rating"]
        .fillna(rating_mode)
        .astype(int)
    )

    clean_df["category"] = (
        clean_df["category"]
        .fillna("Unknown")
        .replace("", "Unknown")
    )

    clean_df["in_stock"] = (
        clean_df["availability"]
        .astype(str)
        .str.contains(
            "In stock",
            case=False,
            na=False
        )
    )

    clean_df["price_inr"] = (
        clean_df["price_gbp"]
        * GBP_TO_INR
    ).round(2)

    clean_df = clean_df[
        [
            "title",
            "price",
            "star_rating",
            "availability",
            "category",
            "url",
            "price_gbp",
            "rating",
            "in_stock",
            "price_inr",
        ]
    ]

    return clean_df


def validate_pipeline(df):
    if len(df) < 60:
        raise ValueError(
            "Dataset must contain at least 60 books."
        )

    if df["category"].nunique() < 3:
        raise ValueError(
            "Dataset must contain at least 3 categories."
        )

    required_columns = {
        "title",
        "price_gbp",
        "rating",
        "in_stock",
        "category",
        "price_inr",
    }

    if not required_columns.issubset(
        set(df.columns)
    ):
        raise ValueError(
            "Required columns are missing."
        )

    if not pd.api.types.is_bool_dtype(
        df["in_stock"]
    ):
        raise ValueError(
            "in_stock must be Boolean."
        )

    expected_price_inr = (
        df["price_gbp"]
        * GBP_TO_INR
    ).round(2)

    difference = (
        expected_price_inr
        - df["price_inr"]
    ).abs()

    if not difference.lt(
        0.01
    ).all():
        raise ValueError(
            "price_inr conversion is incorrect."
        )

    if not df["rating"].isin(
        [1, 2, 3, 4, 5]
    ).all():
        raise ValueError(
            "Ratings must be integers from 1 to 5."
        )

    if df["title"].isna().any():
        raise ValueError(
            "Book titles cannot be missing."
        )

    if df["category"].isna().any():
        raise ValueError(
            "Book categories cannot be missing."
        )

    if (
        df["price_gbp"] <= 0
    ).any():
        raise ValueError(
            "Book prices must be positive."
        )


def setup_database(df):
    connection = sqlite3.connect(
        DB_PATH
    )

    cursor = connection.cursor()

    cursor.execute(
        "PRAGMA foreign_keys = ON"
    )

    cursor.execute(
        "DROP TABLE IF EXISTS books"
    )

    cursor.execute(
        "DROP TABLE IF EXISTS categories"
    )

    cursor.execute(
        """
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT NOT NULL UNIQUE
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            rating INTEGER NOT NULL,
            in_stock INTEGER NOT NULL,
            category_id INTEGER NOT NULL,
            price_inr REAL NOT NULL,
            url TEXT,
            FOREIGN KEY (category_id)
                REFERENCES categories(category_id)
        )
        """
    )

    categories = sorted(
        df["category"]
        .dropna()
        .unique()
        .tolist()
    )

    cursor.executemany(
        """
        INSERT INTO categories (
            category_name
        )
        VALUES (?)
        """,
        [
            (category,)
            for category in categories
        ],
    )

    category_rows = cursor.execute(
        """
        SELECT
            category_id,
            category_name
        FROM categories
        """
    ).fetchall()

    category_map = {
        name: category_id
        for category_id, name
        in category_rows
    }

    book_rows = []

    for _, row in df.iterrows():
        book_rows.append(
            (
                row["title"],
                float(row["price_gbp"]),
                int(row["rating"]),
                int(row["in_stock"]),
                category_map[
                    row["category"]
                ],
                float(row["price_inr"]),
                row["url"],
            )
        )

    cursor.executemany(
        """
        INSERT INTO books (
            title,
            price_gbp,
            rating,
            in_stock,
            category_id,
            price_inr,
            url
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        book_rows,
    )

    connection.commit()

    return connection


def run_queries(connection):
    queries = {
        "SELECT": """
            SELECT
                title,
                price_gbp,
                rating
            FROM books
        """,

        "WHERE": """
            SELECT
                title,
                price_gbp
            FROM books
            WHERE price_gbp > 20
        """,

        "ORDER BY": """
            SELECT
                title,
                price_gbp
            FROM books
            ORDER BY price_gbp DESC
        """,

        "LIMIT": """
            SELECT
                title,
                price_gbp
            FROM books
            ORDER BY price_gbp DESC
            LIMIT 5
        """,

        "DISTINCT": """
            SELECT DISTINCT
                rating
            FROM books
            ORDER BY rating
        """,

        "IN_BETWEEN": """
            SELECT
                title,
                price_gbp,
                rating
            FROM books
            WHERE rating IN (4, 5)
              AND price_gbp BETWEEN 10 AND 30
        """,

        "JOIN": """
            SELECT
                b.title,
                b.price_gbp,
                b.rating,
                c.category_name
            FROM books b
            JOIN categories c
                ON b.category_id =
                   c.category_id
        """,

        "CATEGORY_COUNT": """
            SELECT
                c.category_name,
                COUNT(*) AS book_count
            FROM books b
            JOIN categories c
                ON b.category_id =
                   c.category_id
            GROUP BY c.category_name
            ORDER BY book_count DESC
        """,
    }

    with open(
        SQL_RESULTS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        for name, query in queries.items():

            result = pd.read_sql_query(
                query,
                connection
            )

            print()
            print(
                f"SQL QUERY: {name}"
            )

            print(
                result.head(10)
                .to_string(index=False)
            )

            file.write(
                f"\n{'=' * 60}\n"
            )

            file.write(
                f"SQL QUERY: {name}\n"
            )

            file.write(
                query.strip()
            )

            file.write(
                "\n\nRESULT:\n"
            )

            file.write(
                result.head(10)
                .to_string(index=False)
            )

            file.write("\n")


def pandas_equivalents(connection):
    top_books_query = """
        SELECT
            title,
            price_gbp,
            rating
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 5
    """

    top_books = pd.read_sql(
        top_books_query,
        connection
    )

    print()
    print(
        "PANDAS read_sql RESULT:"
    )

    print(
        top_books.to_string(
            index=False
        )
    )

    category_query = """
        SELECT
            category_id,
            category_name
        FROM categories
    """

    category_df = pd.read_sql(
        category_query,
        connection
    )

    book_query = """
        SELECT
            book_id,
            title,
            price_gbp,
            rating,
            in_stock,
            category_id,
            price_inr
        FROM books
    """

    book_df = pd.read_sql(
        book_query,
        connection
    )

    merged = pd.merge(
        book_df,
        category_df,
        on="category_id",
        how="inner"
    )

    print()
    print(
        "PANDAS MERGE JOIN RESULT:"
    )

    print(
        merged[
            [
                "title",
                "price_gbp",
                "rating",
                "category_name",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


def validate_database(connection):
    cursor = connection.cursor()

    foreign_keys = cursor.execute(
        """
        PRAGMA foreign_key_list(books)
        """
    ).fetchall()

    if len(foreign_keys) == 0:
        raise ValueError(
            "Books table must have a foreign key."
        )

    category_count = cursor.execute(
        """
        SELECT COUNT(*)
        FROM categories
        """
    ).fetchone()[0]

    book_count = cursor.execute(
        """
        SELECT COUNT(*)
        FROM books
        """
    ).fetchone()[0]

    if category_count < 3:
        raise ValueError(
            "Database must contain at least 3 categories."
        )

    if book_count < 60:
        raise ValueError(
            "Database must contain at least 60 books."
        )

    orphan_count = cursor.execute(
        """
        SELECT COUNT(*)
        FROM books b
        LEFT JOIN categories c
            ON b.category_id =
               c.category_id
        WHERE c.category_id IS NULL
        """
    ).fetchone()[0]

    if orphan_count != 0:
        raise ValueError(
            "Books table contains invalid category references."
        )


def main():
    print(
        "Starting Books to Scrape pipeline..."
    )

    print()

    df = scrape_books()

    print()

    print(
        f"Scraped {len(df)} books."
    )

    df.to_csv(
        RAW_CSV_PATH,
        index=False,
        encoding="utf-8"
    )

    clean_df = clean_data(
        df
    )

    print()

    print(
        f"Cleaned dataset contains "
        f"{len(clean_df)} books."
    )

    print()

    print(
        "Categories:"
    )

    print(
        clean_df["category"]
        .value_counts()
    )

    validate_pipeline(
        clean_df
    )

    print()

    print(
        "Pipeline data validation passed."
    )

    clean_df.to_csv(
        CLEAN_CSV_PATH,
        index=False,
        encoding="utf-8"
    )

    connection = setup_database(
        clean_df
    )

    validate_database(
        connection
    )

    print()

    print(
        "Database validation passed."
    )

    run_queries(
        connection
    )

    pandas_equivalents(
        connection
    )

    connection.close()

    print()

    print(
        "Pipeline completed successfully."
    )

    print(
        f"Database: {DB_PATH}"
    )

    print(
        f"Raw CSV: {RAW_CSV_PATH}"
    )

    print(
        f"Clean CSV: {CLEAN_CSV_PATH}"
    )

    print(
        f"SQL results: {SQL_RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()