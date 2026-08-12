"""
End-to-End CLI Pipeline for AI Labor Market Polarization Analysis.

Usage:
    python run_analysis.py --dataset organic --input data/raw/linkedin_jobs.csv
    python run_analysis.py --dataset synthetic --input data/raw/ai_jobs_2026.csv
"""

import argparse
import os
import sys
import pandas as pd

from src.config import (
    ARCHETYPE_LABELS_ORGANIC,
    ARCHETYPE_LABELS_SYNTHETIC,
    DEFAULT_K_ORGANIC,
    DEFAULT_K_SYNTHETIC,
)
from src.modeling import (
    compute_elbow_curve,
    extract_tfidf_features,
    fit_kmeans_model,
    project_pca_2d,
)
from src.preprocessing import clean_job_description, categorize_location
from src.visualization import (
    plot_elbow_curve,
    plot_pca_clusters,
    plot_top_keywords_per_cluster,
)


def run_pipeline(dataset_type: str, file_path: str, output_dir: str = "assets") -> None:
    print(f"==================================================")
    print(f"  AI Labor Market Analysis: {dataset_type.upper()} DATASET")
    print(f"==================================================")
    
    if not os.path.exists(file_path):
        print(f"[-] Error: File not found at {file_path}")
        print(f"[*] Please check data/README.md for download instructions.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Ingestion
    print(f"[1/5] Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"      Initial shape: {df.shape}")

    # Column identification
    text_col = None
    for candidate in ["Job Description", "job_description", "description", "Job_Description"]:
        if candidate in df.columns:
            text_col = candidate
            break
            
    if not text_col:
        print(f"[-] Error: Could not locate Job Description column in {df.columns.tolist()}")
        sys.exit(1)

    df = df.dropna(subset=[text_col]).copy()
    print(f"      Shape after dropping nulls: {df.shape}")

    # 2. Preprocessing
    print(f"[2/5] Cleaning and tokenizing text corpus...")
    df["clean_desc"] = df[text_col].apply(clean_job_description)

    # 3. TF-IDF Feature Extraction
    print(f"[3/5] Extracting TF-IDF features (unigrams & bigrams)...")
    X, vectorizer = extract_tfidf_features(df["clean_desc"])
    print(f"      TF-IDF Matrix Shape: {X.shape}")

    # 4. Inertia & Clustering
    k = DEFAULT_K_ORGANIC if dataset_type == "organic" else DEFAULT_K_SYNTHETIC
    archetypes = ARCHETYPE_LABELS_ORGANIC if dataset_type == "organic" else ARCHETYPE_LABELS_SYNTHETIC
    
    print(f"[4/5] Computing Elbow curve and fitting K-Means (k={k})...")
    inertias = compute_elbow_curve(X, max_k=8)
    elbow_out = os.path.join(output_dir, f"elbow_{dataset_type}.png")
    plot_elbow_curve(inertias, optimal_k=k, save_path=elbow_out)
    print(f"      Saved elbow plot -> {elbow_out}")

    kmeans, labels = fit_kmeans_model(X, n_clusters=k)
    df["cluster"] = labels

    # 5. Dimensionality Reduction & Visualization
    print(f"[5/5] Projecting to 2D PCA & plotting clusters...")
    coords_2d = project_pca_2d(X)
    
    pca_out = os.path.join(output_dir, f"pca_{dataset_type}.png")
    plot_pca_clusters(
        coords_2d,
        labels,
        title=f"PCA Cluster Separation - {dataset_type.capitalize()} Dataset",
        archetype_names=archetypes,
        save_path=pca_out,
    )
    print(f"      Saved PCA plot -> {pca_out}")

    words_out = os.path.join(output_dir, f"top_words_{dataset_type}.png")
    feature_names = vectorizer.get_feature_names_out().tolist()
    plot_top_keywords_per_cluster(
        kmeans,
        feature_names,
        n_top_words=8,
        archetype_names=archetypes,
        save_path=words_out,
    )
    print(f"      Saved keywords plot -> {words_out}")

    print(f"\n[+] Analysis complete! Visualizations generated in '{output_dir}/'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Labor Market NLP & Clustering Pipeline")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["organic", "synthetic"],
        default="organic",
        help="Dataset type to analyze ('organic' or 'synthetic')",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/linkedin_jobs.csv",
        help="Path to CSV dataset file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="assets",
        help="Directory to save generated figures",
    )
    args = parser.parse_args()
    run_pipeline(args.dataset, args.input, args.output_dir)
