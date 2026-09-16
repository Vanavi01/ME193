'''
Trains a Random Forest on the landmarks recorded by record_gestures.py and
compares it against Google's gesture_recognizer.task predictions stored in
the same CSV.

Reports accuracy under two evaluations:
  - naive: a single random-row train/test split. Frames from the same
    recording session are highly correlated, so some near-duplicates of
    the test set leak into training and inflate accuracy.
  - leave-one-session-out (LOSO) cross-validation: each fold holds out
    exactly one session per label (a percentage split can't guarantee
    every label is represented once test sessions get outnumbered by
    classes), so held-out rows always come from a session the model has
    never seen. Reports mean +/- stdev accuracy across folds, and a
    confusion matrix pooled from every fold's held-out predictions.

Usage:
    python train_gestures.py [gesture_landmarks.csv] [--model-out gesture_model.joblib]
'''

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

NUM_LANDMARKS = 21
LANDMARK_COLUMNS = [f"{axis}{i}" for i in range(NUM_LANDMARKS) for axis in ("x", "y", "z")]


def fit_and_score(train_df, test_df, random_state):
    X_train, y_train = train_df[LANDMARK_COLUMNS], train_df["true_label"]
    X_test, y_test = test_df[LANDMARK_COLUMNS], test_df["true_label"]

    clf = RandomForestClassifier(random_state=random_state)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    our_acc = accuracy_score(y_test, preds)
    google_acc = accuracy_score(y_test, test_df["google_label"])
    return preds, y_test, our_acc, google_acc


def leave_one_session_out_folds(df):
    """Yields (fold_index, train_df, test_df). Each fold holds out exactly
    one session per label, so no fold's test rows share a session with
    anything it trained on. The number of folds is capped by whichever
    label has the fewest recorded sessions."""
    sessions_by_label = {
        label: sorted(group["session"].unique())
        for label, group in df.groupby("true_label")
    }
    n_folds = min(len(sessions) for sessions in sessions_by_label.values())

    for fold in range(n_folds):
        test_sessions = {sessions[fold] for sessions in sessions_by_label.values()}
        test_df = df[df["session"].isin(test_sessions)]
        train_df = df[~df["session"].isin(test_sessions)]
        yield fold, train_df, test_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", default="gesture_landmarks.csv")
    parser.add_argument("--model-out", default="gesture_model.joblib")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test fraction for the naive random-row split")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df["google_label"] = df["google_label"].fillna("None")

    miss_rate = df.groupby("true_label")["google_label"].apply(lambda s: (s == "None").mean() * 100)
    print("Google gesture_recognizer.task miss rate per gesture (frames it returned no gesture for):")
    for label, pct in miss_rate.items():
        print(f"  {label}: {pct:.1f}%")
    print()

    naive_train, naive_test = train_test_split(
        df, test_size=args.test_size, random_state=args.random_state, stratify=df["true_label"],
    )
    _, _, naive_our_acc, naive_google_acc = fit_and_score(naive_train, naive_test, args.random_state)

    fold_our_accs, fold_google_accs = [], []
    pooled_y_test, pooled_preds = [], []
    n_folds = 0
    for fold, train_df, test_df in leave_one_session_out_folds(df):
        n_folds += 1
        preds, y_test, our_acc, google_acc = fit_and_score(train_df, test_df, args.random_state)
        fold_our_accs.append(our_acc)
        fold_google_accs.append(google_acc)
        pooled_y_test.append(y_test)
        pooled_preds.append(preds)

    fold_our_accs = np.array(fold_our_accs)
    fold_google_accs = np.array(fold_google_accs)
    y_test_pooled = pd.concat(pooled_y_test)
    preds_pooled = np.concatenate(pooled_preds)

    print(f"Rows: {len(df)}   Sessions: {df['session'].nunique()}   LOSO folds: {n_folds}")
    print()
    print(f"Naive random-row split       (test rows={len(naive_test)}):")
    print(f"  Google accuracy: {naive_google_acc:.3f}")
    print(f"  Random Forest:   {naive_our_acc:.3f}")
    print()
    print(f"Leave-one-session-out CV     ({n_folds} folds, one held-out session per label each):")
    print(f"  Google accuracy: {fold_google_accs.mean():.3f} +/- {fold_google_accs.std():.3f}   {np.round(fold_google_accs, 3).tolist()}")
    print(f"  Random Forest:   {fold_our_accs.mean():.3f} +/- {fold_our_accs.std():.3f}   {np.round(fold_our_accs, 3).tolist()}")
    print()
    print(f"Leakage inflation (Random Forest, naive - LOSO mean): {naive_our_acc - fold_our_accs.mean():+.3f}")

    print("\nClassification report (Random Forest, pooled across LOSO folds):")
    print(classification_report(y_test_pooled, preds_pooled))

    labels = sorted(df["true_label"].unique())
    cm = confusion_matrix(y_test_pooled, preds_pooled, labels=labels)
    print("Confusion matrix (Random Forest, pooled across LOSO folds), rows=true, cols=predicted:")
    print(pd.DataFrame(cm, index=labels, columns=labels))

    final_clf = RandomForestClassifier(random_state=args.random_state)
    final_clf.fit(df[LANDMARK_COLUMNS], df["true_label"])
    joblib.dump(final_clf, args.model_out)
    print(f"\nSaved model (retrained on all {len(df)} rows; accuracy above is from LOSO CV) to {args.model_out}")


if __name__ == "__main__":
    main()
