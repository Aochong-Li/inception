"""
Train fasttext model for safety alert classification.
Handles imbalanced dataset (92% label 0, 8% label 1).
"""
import os
import pandas as pd
import fasttext
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_curve
import matplotlib.pyplot as plt
from typing import Tuple

def _patched_predict(self, text, k=1, threshold=0.0, on_unicode_error='strict'):
    """Patched predict method that works with numpy 2.0+"""
    predictions = self.f.predict(text, k, threshold, on_unicode_error)
    if len(predictions) == 0:
        return ([], [])
    # predictions is list of tuples (prob, label)
    probs, labels = zip(*predictions)
    return tuple(labels), np.asarray(probs)

fasttext.FastText._FastText.predict = _patched_predict

def format_for_fasttext(text: str, label: int) -> str:
    """Format a single sample for fasttext training."""
    # Clean text: remove newlines and extra spaces
    cleaned_text = ' '.join(text.split())
    return f"__label__{label} {cleaned_text}"


def prepare_data(
    input_pickle: str,
    output_dir: str,
    test_size: float = 0.2,
    oversample_minority: bool = True,
    oversample_ratio: float = 0.3
) -> Tuple[str, str, pd.DataFrame, pd.DataFrame]:
    """
    Load raw data and prepare fasttext format files.

    Args:
        input_pickle: Path to raw_data.pickle
        output_dir: Directory to save train/test files
        test_size: Fraction of data for testing
        oversample_minority: Whether to oversample minority class
        oversample_ratio: Target ratio for minority class (0.3 = 30% of final dataset)

    Returns:
        train_file, test_file, train_df, test_df
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_pickle(input_pickle)
    print(f"Original class ratio: {df['label'].value_counts(normalize=True).to_dict()}")
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['label'],
        random_state=42
    )

    if oversample_minority:
        train_df = oversample_minority_class(train_df, target_ratio=oversample_ratio)
        print(f"\nAfter oversampling minority class:")
        print(train_df['label'].value_counts())
        print(f"New class ratio: {train_df['label'].value_counts(normalize=True).to_dict()}")

    train_file = os.path.join(output_dir, 'train.txt')
    with open(train_file, 'w', encoding='utf-8') as f:
        for _, row in train_df.iterrows():
            f.write(format_for_fasttext(row['input_text'], row['label']) + '\n')

    test_file = os.path.join(output_dir, 'test.txt')
    with open(test_file, 'w', encoding='utf-8') as f:
        for _, row in test_df.iterrows():
            f.write(format_for_fasttext(row['input_text'], row['label']) + '\n')

    print(f"\nSaved train file: {train_file} ({len(train_df)} samples)")
    print(f"Saved test file: {test_file} ({len(test_df)} samples)")

    return train_file, test_file, train_df, test_df


def oversample_minority_class(df: pd.DataFrame, target_ratio: float = 0.3) -> pd.DataFrame:
    """
    Oversample minority class to achieve target ratio.

    Args:
        df: DataFrame with 'label' column
        target_ratio: Target proportion for minority class (e.g., 0.3 = 30%)

    Returns:
        Oversampled DataFrame
    """
    # Identify minority and majority classes
    label_counts = df['label'].value_counts()
    minority_label = label_counts.idxmin()
    majority_label = label_counts.idxmax()

    minority_df = df[df['label'] == minority_label]
    majority_df = df[df['label'] == majority_label]

    # Calculate how many minority samples we need
    n_majority = len(majority_df)
    n_minority_needed = int(n_majority * target_ratio / (1 - target_ratio))

    # Oversample with replacement
    minority_oversampled = minority_df.sample(
        n=n_minority_needed,
        replace=True,
        random_state=42
    )

    # Combine and shuffle
    result = pd.concat([majority_df, minority_oversampled], ignore_index=True)
    result = result.sample(frac=1, random_state=42).reset_index(drop=True)

    return result


def train_model(
    train_file: str,
    model_output_path: str,
    lr: float = 0.1,
    epoch: int = 50,
    wordNgrams: int = 2,
    dim: int = 500,
    loss: str = 'softmax',
    verbose: int = 2,
    vector_file: str = None
) -> fasttext.FastText._FastText:
    """
    Train fasttext model.

    Args:
        train_file: Path to training file
        model_output_path: Where to save the model
        lr: Learning rate
        epoch: Number of epochs
        wordNgrams: Max length of word ngrams
        dim: Size of word vectors
        loss: Loss function ('softmax', 'hs', 'ova')
        verbose: Verbosity level

    Returns:
        Trained model
    """
    print(f"\nTraining fasttext model...")
    print(f"Parameters: lr={lr}, epoch={epoch}, wordNgrams={wordNgrams}, dim={dim}, loss={loss}")

    model = fasttext.train_supervised(
        input=train_file,
        lr=lr,
        epoch=epoch,
        wordNgrams=wordNgrams,
        dim=dim,
        loss=loss,
        verbose=verbose,
        minCount=3
    )

    # Save model
    model.save_model(model_output_path)
    print(f"Model saved to: {model_output_path}")

    return model


def evaluate_model(
    model: fasttext.FastText._FastText,
    test_df: pd.DataFrame,
    threshold: float = 0.5
) -> dict:
    """
    Evaluate model on test set.

    Args:
        model: Trained fasttext model
        test_df: Test DataFrame with 'input_text' and 'label' columns
        threshold: Classification threshold for label 1

    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    # Get predictions
    y_true = test_df['label'].values
    predictions = [model.predict(text) for text in test_df['input_text']]

    # Extract labels and probabilities
    y_pred_labels = [int(pred[0][0].replace('__label__', '')) for pred in predictions]
    y_pred_probs = [pred[1][0] for pred in predictions]

    # Apply threshold for binary classification
    y_pred_thresh = [1 if (label == 1 and prob >= threshold) else 0
                     for label, prob in zip(y_pred_labels, y_pred_probs)]

    # Classification report
    print(f"\nClassification Report (threshold={threshold}):")
    print(classification_report(y_true, y_pred_thresh, target_names=['Label 0', 'Label 1']))

    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred_thresh)
    print(cm)
    print(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")

    # F1 scores
    f1_macro = f1_score(y_true, y_pred_thresh, average='macro')
    f1_weighted = f1_score(y_true, y_pred_thresh, average='weighted')
    f1_label1 = f1_score(y_true, y_pred_thresh, pos_label=1)

    print(f"\nF1 Scores:")
    print(f"  Macro F1: {f1_macro:.4f}")
    print(f"  Weighted F1: {f1_weighted:.4f}")
    print(f"  Label 1 F1: {f1_label1:.4f}")

    # Find optimal threshold using PR curve
    # Extract probabilities for label 1
    y_scores = []
    for text in test_df['input_text']:
        pred = model.predict(text)
        label = int(pred[0][0].replace('__label__', ''))
        prob = pred[1][0]
        # If prediction is label 0, probability for label 1 is (1 - prob)
        y_scores.append(prob if label == 1 else 1 - prob)

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_threshold_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else threshold

    print(f"\nOptimal threshold based on F1: {best_threshold:.4f}")
    print(f"  Precision at optimal: {precision[best_threshold_idx]:.4f}")
    print(f"  Recall at optimal: {recall[best_threshold_idx]:.4f}")
    print(f"  F1 at optimal: {f1_scores[best_threshold_idx]:.4f}")

    return {
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_label1': f1_label1,
        'best_threshold': best_threshold,
        'confusion_matrix': cm
    }


if __name__ == "__main__":
    # Configuration
    INPUT_PICKLE = "./fasttext_models/train/data/raw_data.pickle"
    OUTPUT_DIR = "./fasttext_models/train/data"
    MODEL_OUTPUT = "./fasttext_models/refusal_model_softmax.bin"

    # Hyperparameters
    TEST_SIZE = 0.1
    OVERSAMPLE = True
    OVERSAMPLE_RATIO = 0.5

    # Training parameters
    LEARNING_RATE = 0.5
    EPOCHS = 50
    WORD_NGRAMS = 3
    DIM = 1000
    LOSS = 'softmax'
    VECTOR_FILE = None

    # Prepare data
    train_file, test_file, train_df, test_df = prepare_data(
        input_pickle=INPUT_PICKLE,
        output_dir=OUTPUT_DIR,
        test_size=TEST_SIZE,
        oversample_minority=OVERSAMPLE,
        oversample_ratio=OVERSAMPLE_RATIO
    )

    # Train model
    model = train_model(
        train_file=train_file,
        model_output_path=MODEL_OUTPUT,
        lr=LEARNING_RATE,
        epoch=EPOCHS,
        wordNgrams=WORD_NGRAMS,
        dim=DIM,
        loss=LOSS,
        vector_file=VECTOR_FILE
    )

    # Evaluate model
    results = evaluate_model(model, test_df, threshold=0.5)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Model saved to: {MODEL_OUTPUT}")
    print(f"Recommended threshold: {results['best_threshold']:.4f}")
