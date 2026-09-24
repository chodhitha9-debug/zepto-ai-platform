
import sqlite3
import pandas as pd
from textblob import TextBlob


# --- 1. Tool Implementations ---
def check_inventory(product_name: str) -> str:
    """Queries the SQLite database created in Module 1 to check book inventory."""
    conn = sqlite3.connect("data_pipeline/zepto_catalog.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT title, price_inr, in_stock FROM books WHERE title LIKE ?",
        (f"%{product_name}%",),
    )
    results = cursor.fetchall()
    conn.close()

    if not results:
        return f"No products found matching '{product_name}'."

    res_str = ""
    for title, price, stock in results:
        status = "In Stock" if stock == 1 else "Out of Stock"
        res_str += f"- {title}: ₹{price} ({status})\n"
    return res_str.strip()


def calculate_discount(price: float, discount_pct: float) -> float:
    """Calculates final price after applying discount percentage."""
    discount_amount = price * (discount_pct / 100.0)
    return round(price - discount_amount, 2)


# --- 2. Sentiment Analysis System ---
def analyze_sentiment(user_message: str) -> dict:
    """Analyzes customer message sentiment to adjust tone and escalation status."""
    analysis = TextBlob(user_message)
    polarity = analysis.sentiment.polarity

    if polarity < -0.2:
        sentiment = "Frustrated"
        tone_instruction = "Apologetic, empathetic, and highly priority-focused."
        escalate = True
    elif polarity > 0.2:
        sentiment = "Positive"
        tone_instruction = "Warm, enthusiastic, and helpful."
        escalate = False
    else:
        sentiment = "Neutral"
        tone_instruction = "Professional, concise, and direct."
        escalate = False

    return {
        "sentiment": sentiment,
        "polarity_score": round(polarity, 2),
        "tone_instruction": tone_instruction,
        "escalate": escalate,
    }


# --- 3. Assistant Orchestration Demo ---
def run_support_assistant(user_message: str, product_query: str = None, price: float = None, discount: float = None):
    print("=" * 50)
    print(f"User Message: '{user_message}'")

    # Step A: Sentiment Analysis
    sentiment_info = analyze_sentiment(user_message)
    print(f"\n[Sentiment Analysis]")
    print(f"Detected Sentiment: {sentiment_info['sentiment']} (Score: {sentiment_info['polarity_score']})")
    print(f"Tone Strategy: {sentiment_info['tone_instruction']}")
    print(f"Escalation Flag: {sentiment_info['escalate']}")

    # Step B: Tool Execution
    print(f"\n[Tool Execution]")
    if product_query:
        inv_result = check_inventory(product_query)
        print(f"Inventory Check Tool Result:\n{inv_result}")

    if price is not None and discount is not None:
        disc_result = calculate_discount(price, discount)
        print(f"Discount Tool Result: Original ₹{price} with {discount}% off -> Final: ₹{disc_result}")

    print("=" * 50 + "\n")


if __name__ == "__main__":
    # Test Case 1: Neutral inquiry with inventory check
    run_support_assistant(
        user_message="Hello, do you have any books about Sapiens in stock?",
        product_query="Sapiens"
    )

    # Test Case 2: Frustrated user with discount calculation
    run_support_assistant(
        user_message="My order is taking forever! Can I get a 15% discount on this 500 rupees book?",
        price=500.0,
        discount=15.0
    )