import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def run_eda():
    os.makedirs("analytics/charts", exist_ok=True)

    source_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
    csv_path = "analytics/titanic.csv"

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print("Loaded Titanic dataset from analytics/titanic.csv")
    else:
        df = pd.read_csv(source_url)
        df.to_csv(csv_path, index=False)
        print("Downloaded Titanic dataset and saved to analytics/titanic.csv")

    print("\n--- Dataset Info ---")
    print(df.info())

    print("\n--- Dataset Shape ---")
    print(df.shape)

    print("\n--- Descriptive Statistics ---")
    print(df.describe(include="all"))

    print("\n--- Missing Values Before Cleaning ---")
    missing_counts = df.isnull().sum()
    missing_percent = df.isnull().mean() * 100

    missing_table = pd.DataFrame(
        {
            "Missing Count": missing_counts,
            "Missing Percentage": missing_percent,
        }
    )

    print(missing_table[missing_table["Missing Count"] > 0])

    print("\n--- Missing Value Treatment Strategy ---")

    for column in df.columns:
        percentage = df[column].isnull().mean() * 100

        if percentage == 0:
            continue

        if percentage < 5:
            print(
                f"{column}: {percentage:.2f}% missing -> "
                "drop rows with missing values"
            )
        elif percentage <= 30:
            print(
                f"{column}: {percentage:.2f}% missing -> "
                "impute using an appropriate strategy"
            )
        else:
            print(
                f"{column}: {percentage:.2f}% missing -> "
                "drop column because missingness is very high"
            )

    if "deck" in df.columns:
        df = df.drop(columns=["deck"])

    if "age" in df.columns:
        df["age"] = df["age"].fillna(df["age"].median())

    if "embarked" in df.columns:
        df = df.dropna(subset=["embarked"])

    if "embark_town" in df.columns:
        df = df.dropna(subset=["embark_town"])

    print("\n--- Cleaning Decisions ---")
    print("deck: dropped because approximately 77% of values were missing.")
    print("age: median imputation because approximately 20% of values were missing.")
    print("embarked: rows dropped because missingness was below 5%.")
    print("embark_town: rows dropped because missingness was below 5%.")

    print("\n--- Missing Values After Cleaning ---")
    print(df.isnull().sum())

    print("\n--- Cleaned Dataset Shape ---")
    print(df.shape)

    age_q1 = df["age"].quantile(0.25)
    age_q3 = df["age"].quantile(0.75)
    age_iqr = age_q3 - age_q1

    age_lower = age_q1 - 1.5 * age_iqr
    age_upper = age_q3 + 1.5 * age_iqr

    age_outliers = df[
        (df["age"] < age_lower) | (df["age"] > age_upper)
    ]

    fare_q1 = df["fare"].quantile(0.25)
    fare_q3 = df["fare"].quantile(0.75)
    fare_iqr = fare_q3 - fare_q1

    fare_lower = fare_q1 - 1.5 * fare_iqr
    fare_upper = fare_q3 + 1.5 * fare_iqr

    fare_outliers = df[
        (df["fare"] < fare_lower) | (df["fare"] > fare_upper)
    ]

    print("\n--- IQR Outlier Analysis ---")
    print(f"Age outlier count: {len(age_outliers)}")
    print(f"Fare outlier count: {len(fare_outliers)}")

    print("\n--- Fare Statistics ---")
    fare_mean = df["fare"].mean()
    fare_median = df["fare"].median()
    fare_mode = df["fare"].mode().iloc[0]
    fare_skewness = df["fare"].skew()

    print(f"Mean: {fare_mean:.2f}")
    print(f"Median: {fare_median:.2f}")
    print(f"Mode: {fare_mode:.2f}")
    print(f"Skewness: {fare_skewness:.2f}")

    if fare_skewness > 0:
        print(
            "Conclusion: Fare is positively skewed, with a longer "
            "right tail caused by relatively high fares."
        )
    elif fare_skewness < 0:
        print(
            "Conclusion: Fare is negatively skewed, with a longer "
            "left tail."
        )
    else:
        print("Conclusion: Fare is approximately symmetric.")

    plt.figure(figsize=(10, 6))
    sns.histplot(df["age"], kde=True)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("analytics/charts/age_histogram.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df["age"])
    plt.title("Age Boxplot")
    plt.xlabel("Age")
    plt.tight_layout()
    plt.savefig("analytics/charts/age_boxplot.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(df["fare"], kde=True)
    plt.title("Fare Distribution")
    plt.xlabel("Fare")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("analytics/charts/fare_histogram.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df["fare"])
    plt.title("Fare Boxplot")
    plt.xlabel("Fare")
    plt.tight_layout()
    plt.savefig("analytics/charts/fare_boxplot.png", dpi=200)
    plt.close()

    survival_by_sex = (
        df.groupby("sex")["survived"]
        .mean()
        .reset_index()
    )

    print("\n--- Survival Rate by Sex ---")
    print(survival_by_sex)

    survival_by_class = (
        df.groupby("pclass")["survived"]
        .mean()
        .reset_index()
    )

    print("\n--- Survival Rate by Passenger Class ---")
    print(survival_by_class)

    survival_by_sex_class = (
        df.groupby(["sex", "pclass"])["survived"]
        .mean()
        .reset_index()
    )

    print("\n--- Survival Rate by Sex and Passenger Class ---")
    print(survival_by_sex_class)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df,
        x="sex",
        y="survived",
        errorbar=None,
    )
    plt.title("Survival Rate by Sex")
    plt.xlabel("Sex")
    plt.ylabel("Survival Rate")
    plt.tight_layout()
    plt.savefig("analytics/charts/survival_by_sex.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df,
        x="pclass",
        y="survived",
        errorbar=None,
    )
    plt.title("Survival Rate by Passenger Class")
    plt.xlabel("Passenger Class")
    plt.ylabel("Survival Rate")
    plt.tight_layout()
    plt.savefig(
        "analytics/charts/survival_by_class.png",
        dpi=200,
    )
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df,
        x="pclass",
        y="survived",
        hue="sex",
        errorbar=None,
    )
    plt.title("Survival Rate by Sex and Passenger Class")
    plt.xlabel("Passenger Class")
    plt.ylabel("Survival Rate")
    plt.tight_layout()
    plt.savefig(
        "analytics/charts/survival_by_sex_class.png",
        dpi=200,
    )
    plt.close()

    corr_cols = [
        "survived",
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare",
    ]

    correlation_matrix = df[corr_cols].corr()

    print("\n--- Required 6x6 Correlation Matrix ---")
    print(correlation_matrix)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
    )
    plt.title(
        "Correlation Heatmap: Survived, Pclass, Age, SibSp, Parch, Fare"
    )
    plt.tight_layout()
    plt.savefig(
        "analytics/charts/correlation_heatmap.png",
        dpi=200,
    )
    plt.close()

    correlation_pairs = []

    for i in range(len(corr_cols)):
        for j in range(i + 1, len(corr_cols)):
            correlation_pairs.append(
                (
                    corr_cols[i],
                    corr_cols[j],
                    correlation_matrix.iloc[i, j],
                    abs(correlation_matrix.iloc[i, j]),
                )
            )

    correlation_pairs.sort(
        key=lambda x: x[3],
        reverse=True,
    )

    print("\n--- Two Strongest Absolute Correlations ---")

    for pair in correlation_pairs[:2]:
        print(
            f"{pair[0]} vs {pair[1]}: "
            f"{pair[2]:.4f} "
            f"(absolute correlation {pair[3]:.4f})"
        )

    standardization_data = df[["age", "fare"]].copy()

    scaler_means = standardization_data.mean()
    scaler_stds = standardization_data.std()

    standardized = (
        standardization_data - scaler_means
    ) / scaler_stds

    print("\n--- Standardization Check ---")
    print("Standardized means:")
    print(standardized.mean())
    print("Standardized standard deviations:")
    print(standardized.std())

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df,
        x="age",
        y="fare",
        hue="survived",
        style="sex",
        alpha=0.7,
    )
    plt.title("Age vs Fare by Survival and Sex")
    plt.xlabel("Age")
    plt.ylabel("Fare")
    plt.tight_layout()
    plt.savefig(
        "analytics/charts/multivariate_age_fare.png",
        dpi=200,
    )
    plt.close()

    print("\nMultivariate Chart 1 Interpretation:")
    print(
        "The age-versus-fare chart examines two numerical variables "
        "simultaneously while separating passengers by survival and sex."
    )
    print(
        "Higher-fare observations are spread across age groups, while "
        "survival status and sex provide additional grouping information."
    )

    plt.figure(figsize=(10, 7))
    sns.boxplot(
        data=df,
        x="pclass",
        y="age",
        hue="sex",
    )
    plt.title("Age Distribution by Passenger Class and Sex")
    plt.xlabel("Passenger Class")
    plt.ylabel("Age")
    plt.tight_layout()
    plt.savefig(
        "analytics/charts/multivariate_age_class_sex.png",
        dpi=200,
    )
    plt.close()

    print("\nMultivariate Chart 2 Interpretation:")
    print(
        "The boxplot compares age distributions across passenger classes "
        "while also separating the observations by sex."
    )
    print(
        "It shows how the age distribution differs between classes and "
        "whether those differences are similar for male and female passengers."
    )

    plt.figure(figsize=(10, 7))
    sns.boxplot(
        data=df,
        x="pclass",
        y="fare",
        hue="survived",
    )
    plt.title("Fare Distribution by Passenger Class and Survival")
    plt.xlabel("Passenger Class")
    plt.ylabel("Fare")
    plt.tight_layout()
    plt.savefig(
        "analytics/charts/multivariate_fare_class_survival.png",
        dpi=200,
    )
    plt.close()

    print("\nMultivariate Chart 3 Interpretation:")
    print(
        "The fare boxplot examines fare differences across passenger "
        "classes while separating passengers by survival status."
    )
    print(
        "Higher passenger classes generally contain higher fares, and "
        "the survival grouping shows how fare distributions overlap "
        "between survivors and non-survivors."
    )

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df,
        x="age",
        y="fare",
        hue="pclass",
        size="survived",
        style="sex",
        alpha=0.7,
    )
    plt.title("Age, Fare, Passenger Class, Survival and Sex")
    plt.xlabel("Age")
    plt.ylabel("Fare")
    plt.tight_layout()
    plt.savefig(
        "analytics/charts/multivariate_age_fare_class_survival.png",
        dpi=200,
    )
    plt.close()

    print("\nMultivariate Chart 4 Interpretation:")
    print(
        "This chart simultaneously represents age and fare while using "
        "passenger class, survival and sex as additional dimensions."
    )
    print(
        "It provides a combined view of how socioeconomic class, fare, "
        "age and survival status interact within the Titanic data."
    )

    plt.figure(figsize=(10, 7))
    sns.boxplot(
        data=df,
        x="sex",
        y="fare",
        hue="survived",
    )
    plt.title("Fare by Sex and Survival")
    plt.xlabel("Sex")
    plt.ylabel("Fare")
    plt.tight_layout()
    plt.savefig(
        "analytics/charts/multivariate_fare_sex_survival.png",
        dpi=200,
    )
    plt.close()

    print("\nAdditional Multivariate Chart Interpretation:")
    print(
        "This chart compares fare distributions by sex while also "
        "separating survivors from non-survivors."
    )
    print(
        "The distribution highlights differences in fare levels within "
        "the sex and survival groups."
    )

    df.to_csv(
        "analytics/titanic_cleaned.csv",
        index=False,
    )

    print("\n--- Output Files ---")
    print("Original modeling dataset: analytics/titanic.csv")
    print("Cleaned EDA dataset: analytics/titanic_cleaned.csv")
    print("Charts: analytics/charts/")

    print("\nEDA completed successfully.")


if __name__ == "__main__":
    run_eda()