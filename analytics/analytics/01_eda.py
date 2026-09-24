import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def run_eda():
    df_raw = sns.load_dataset("titanic")
    df_raw.to_csv("analytics/titanic.csv", index=False)

    df = pd.read_csv("analytics/titanic.csv")

    print("--- Missing Value Percentages ---")
    missing_pct = (df.isnull().sum() / len(df)) * 100
    print(missing_pct[missing_pct > 0])

    df = df.drop(columns=["deck"])
    df["age"] = df["age"].fillna(df["age"].median())
    df = df.dropna(subset=["embarked", "embark_town"])

    for col in ["age", "fare"]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = df[
            (df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))
        ]
        print(f"Outliers in {col}: {len(outliers)}")

    print(
        f"\nFare Metrics -> Mean: {df['fare'].mean():.2f}, Median: {df['fare'].median():.2f}, Mode: {df['fare'].mode()[0]:.2f}"
    )

    print("\n--- Survival Rates ---")
    print("By Sex:\n", df.groupby("sex")["survived"].mean())
    print("\nBy Class:\n", df.groupby("pclass")["survived"].mean())

    corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    corr_matrix = df[corr_cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.tight_layout()
    plt.savefig("analytics/heatmap.png")
    plt.close()

    print("\nEDA completed. Titanic CSV and Correlation Heatmap saved.")


if __name__ == "__main__":
    run_eda()