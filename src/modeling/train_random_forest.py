from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import project_root


def load_dataset(path: Path) -> pd.DataFrame:
    """Load Titanic dataset from CSV or Excel."""
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Minimal preprocessing used in the coursework report."""
    df = df.copy()

    # Drop columns that are unlikely to help / high-cardinality / sparse
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")

    # Missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Encode categoricals
    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    if "Embarked" in df.columns:
        df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    return X, y


def train_and_evaluate(
    data_path: Path,
    test_size: float = 0.10,
    random_state: int = 42,
    n_estimators: int = 100,
) -> dict:
    df = load_dataset(data_path)
    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def save_confusion_matrix(cm, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Did Not Survive", "Survived"],
        yticklabels=["Did Not Survive", "Survived"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    root = project_root()

    # Prefer the Excel file used in the coursework report; fall back to CSV if needed.
    xlsx_path = root / "data" / "raw" / "TitanicSurvival.xlsx"
    csv_path = root / "data" / "raw" / "train.csv"
    data_path = xlsx_path if xlsx_path.exists() else csv_path

    results = train_and_evaluate(data_path)

    print(f"Accuracy: {results['accuracy']:.4f}")
    print("\nConfusion Matrix:")
    cm = results["confusion_matrix"]
    print(cm)

    # Pretty classification report
    from pprint import pprint
    print("\nClassification Report:")
    pprint(results["classification_report"])

    import numpy as np
    save_confusion_matrix(
        cm=np.array(cm),
        out_path=root / "outputs" / "figures" / "confusion_matrix.png",
    )

    # Also write a lightweight JSON metrics file
    (root / "outputs").mkdir(parents=True, exist_ok=True)
    import json
    with open(root / "outputs" / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
