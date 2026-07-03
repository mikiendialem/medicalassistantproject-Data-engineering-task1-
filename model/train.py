"""Train a symptom -> disease classifier.

IMPORTANT: This model is trained on public demo data and is NOT a validated
medical diagnostic tool. Outputs are ranked possibilities for
educational/triage-style demonstration only.
"""
import json
from pathlib import Path

import joblib
import kagglehub
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)

BASE_DIR = Path(__file__).resolve().parent
KAGGLE_DATASET = "choongqianzheng/disease-and-symptoms-dataset"


def resolve_existing_dir(*relative_paths: str) -> Path:
    candidates = [BASE_DIR.joinpath(path) for path in relative_paths]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def find_csv_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.csv"))


def pick_dataset_csvs(csv_files: list[Path]) -> tuple[Path, Path]:
    if not csv_files:
        raise FileNotFoundError("No CSV files found in the dataset download.")

    name_map = {csv.name.lower(): csv for csv in csv_files}
    preferred_train_names = ["training_data.csv", "train.csv", "training.csv"]
    preferred_test_names = ["test_data.csv", "test.csv", "testing.csv"]

    train_file = next((name_map[name] for name in preferred_train_names if name in name_map), None)
    test_file = next((name_map[name] for name in preferred_test_names if name in name_map), None)

    if train_file and test_file:
        return train_file, test_file

    if len(csv_files) == 1:
        return csv_files[0], csv_files[0]

    csv_files = sorted(csv_files, key=lambda path: path.stat().st_size, reverse=True)
    return csv_files[0], csv_files[1]


def normalize_text(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_")


def load_local_prognosis_dataset(train_csv: Path, test_csv: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    train_df = pd.read_csv(train_csv)
    train_df = train_df.drop(columns=[c for c in train_df.columns if c.startswith("Unnamed")])
    train_df = train_df.drop_duplicates().reset_index(drop=True)

    test_df = pd.read_csv(test_csv)
    test_df = test_df.drop(columns=[c for c in test_df.columns if c.startswith("Unnamed")])

    symptom_cols = [c for c in train_df.columns if c != "prognosis"]
    X_train, y_train = train_df[symptom_cols], train_df["prognosis"]
    X_test, y_test = test_df[symptom_cols], test_df["prognosis"]
    return X_train, y_train, X_test, y_test, symptom_cols


def load_kaggle_symptom_dataset(dataset_csv: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    raw_df = pd.read_csv(dataset_csv)
    raw_df.columns = [column.strip() for column in raw_df.columns]

    label_col = "Disease" if "Disease" in raw_df.columns else "prognosis"
    symptom_cols = [column for column in raw_df.columns if column.startswith("Symptom_")]

    if not symptom_cols:
        raise ValueError("Could not find Symptom_* columns in the Kaggle dataset.")

    normalized_symptoms = raw_df[symptom_cols].applymap(
        lambda value: normalize_text(value) if pd.notna(value) else None
    )
    symptom_list = sorted(
        {
            symptom
            for symptom in normalized_symptoms.stack().dropna().tolist()
            if symptom and symptom != "nan"
        }
    )

    feature_rows = []
    for _, row in normalized_symptoms.iterrows():
        row_symptoms = {symptom for symptom in row.tolist() if symptom and symptom != "nan"}
        feature_rows.append([1 if symptom in row_symptoms else 0 for symptom in symptom_list])

    feature_df = pd.DataFrame(feature_rows, columns=symptom_list)
    target_series = raw_df[label_col].astype(str).str.strip()

    X_train, X_test, y_train, y_test = train_test_split(
        feature_df,
        target_series,
        test_size=0.2,
        random_state=42,
        stratify=target_series,
    )
    return X_train, y_train, X_test, y_test, symptom_list


MODEL_DIR = resolve_existing_dir("model", "../model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = resolve_existing_dir("data", "../data")
training_path = DATA_DIR / "training_data.csv"
test_path = DATA_DIR / "test_data.csv"

if training_path.exists() and test_path.exists():
    X_train, y_train, X_test, y_test, symptom_cols = load_local_prognosis_dataset(training_path, test_path)
else:
    print(f"Local dataset not found in {DATA_DIR}, downloading {KAGGLE_DATASET} via kagglehub...")
    dataset_root = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    downloaded_csvs = find_csv_files(dataset_root)
    kaggle_csv, _ = pick_dataset_csvs(downloaded_csvs)
    print(f"Using Kaggle dataset: {kaggle_csv}")
    X_train, y_train, X_test, y_test, symptom_cols = load_kaggle_symptom_dataset(kaggle_csv)

clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    class_weight="balanced",
)

if y_train.nunique() >= 3 and y_train.value_counts().min() >= 3:
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=skf, scoring="accuracy")
    print(f"CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
else:
    print("CV accuracy: skipped because there are not enough samples per class.")

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"\nHeld-out test accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\nClassification report:\n", classification_report(y_test, y_pred, zero_division=0))

importances = pd.Series(clf.feature_importances_, index=symptom_cols).sort_values(ascending=False)
print("\nTop 15 most informative symptoms:\n", importances.head(15))

joblib.dump(clf, MODEL_DIR / "disease_classifier.joblib")
with open(MODEL_DIR / "symptom_list.json", "w") as f:
    json.dump(symptom_cols, f, indent=2)
with open(MODEL_DIR / "disease_list.json", "w") as f:
    json.dump(sorted(y_train.unique().tolist()), f, indent=2)

print(f"\nSaved model to {MODEL_DIR / 'disease_classifier.joblib'}")
print(f"Saved {len(symptom_cols)} symptoms and {y_train.nunique()} diseases")
