# 📊 Datasets Information & Setup

This repository investigates and contrasts two distinct datasets representing the AI and data science labor market.

---

## 1. Dataset A: Organic LinkedIn Market Data (Real-World)
- **Source:** Scraped from LinkedIn Job Postings (Jan - Apr 2026).
- **Kaggle Link:** [LinkedIn Data Jobs Dataset](https://www.kaggle.com/datasets/joykimaiyo18/linkedin-data-jobs-dataset)
- **Sample Size:** ~1,050 unique job postings.
- **Characteristics:** Unstructured, fluid human language with overlapping role definitions reflecting genuine recruitment practices.
- **Recommended File Location:** `data/raw/linkedin_jobs.csv`

---

## 2. Dataset B: Synthetic AI Jobs 2026 (LLM Simulated)
- **Source:** Synthetically generated benchmark dataset.
- **Kaggle Link:** [AI Jobs Dataset 2026](https://www.kaggle.com/datasets/m0sm71/ai-jobs-dataset-2026)
- **Sample Size:** >50,000 observations.
- **Characteristics:** Highly standardized text, rigid parallel PCA cluster geometries ("cigar" shapes), and decoupled uniform wage/experience distributions.
- **Recommended File Location:** `data/raw/ai_jobs_2026.csv`

---

## 🛠️ Quick Data Setup

1. Download the CSV files from the Kaggle links above.
2. Place them into the `data/raw/` directory:
   ```bash
   mkdir -p data/raw
   # Copy your downloaded CSVs here
   ```
