"""
Configuration settings, hyperparameters, and domain-specific stop words
for the AI Labor Market Polarization analysis.
"""

from typing import List

# Random Seed for reproducibility
RANDOM_STATE: int = 42

# TF-IDF Feature Extraction Parameters
TFIDF_MAX_FEATURES: int = 1000
TFIDF_NGRAM_RANGE: tuple = (1, 2)
TFIDF_MIN_DF: int = 2

# K-Means Clustering Defaults
DEFAULT_K_ORGANIC: int = 5
DEFAULT_K_SYNTHETIC: int = 3
KMEANS_N_INIT: int = 10

# Extended Custom Stop Words (Removing HR Boilerplate, Equal Opportunity Statements, and Generic Terms)
CUSTOM_STOP_WORDS: List[str] = [
    # General Data & Role terms
    "data", "scientist", "science", "learning", "machine", "analytics", "analyst", "role", "job", "work",
    "engineer", "engineering", "experience", "years", "skills", "team", "working", "responsibilities",
    
    # Generic requirements & Soft skills
    "requirements", "qualifications", "required", "preferred", "ability", "strong", "knowledge",
    "opportunity", "equal", "status", "sexual", "orientation", "gender", "race", "color", "religion",
    "national", "origin", "disability", "protected", "veteran", "affirmative", "action", "employer",
    
    # Corporate Boilerplate
    "company", "business", "support", "help", "join", "us", "environment", "growth", "benefits",
    "including", "must", "well", "plus", "understanding", "degree", "bachelor", "master", "phd",
    "proficient", "familiarity", "demonstrated", "hands", "track", "record", "fast", "paced",
    "self", "starter", "communication", "written", "verbal", "interpersonal", "collaborate",
    "candidate", "position", "apply", "successful", "ideal", "level", "provide", "solutions",
    "development", "projects", "new", "technologies", "tools", "systems", "processes"
]

# Cluster Label Annotations
ARCHETYPE_LABELS_ORGANIC = {
    0: "Corporate / Large Scale Enterprise",
    1: "Applied Research & Strategic AI",
    2: "Big Tech / Organizational Core",
    3: "Legacy BI & SQL Analytics (Data Proletariat)",
    4: "MLOps & AI Infrastructure Architect (Technocratic Elite)"
}

ARCHETYPE_LABELS_SYNTHETIC = {
    0: "MLOps & AI Architect",
    1: "AI Software Engineering",
    2: "Applied AI & Research"
}
