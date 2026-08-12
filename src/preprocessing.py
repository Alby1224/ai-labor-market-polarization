"""
Text Preprocessing, NLP cleaning utilities, and metadata parsers.
"""

import re
from typing import List, Optional, Set
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.config import CUSTOM_STOP_WORDS


def get_combined_stop_words() -> List[str]:
    """
    Combines Scikit-Learn's English stop words with domain-specific HR & job boilerplate.

    Returns:
        List[str]: Complete list of stop words for TF-IDF vectorization.
    """
    combined: Set[str] = set(ENGLISH_STOP_WORDS).union(set(CUSTOM_STOP_WORDS))
    return list(combined)


def clean_job_description(text: str) -> str:
    """
    Standard NLP cleaning pipeline for raw job postings:
    - Lowercase conversion
    - URL and email removal
    - Non-alphabetic character stripping
    - Whitespace normalization

    Args:
        text (str): Raw job description text.

    Returns:
        str: Cleaned text ready for tokenization and vectorization.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove URLs and Emails
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text, flags=re.MULTILINE)
    text = re.sub(r"\S+@\S+", " ", text)
    
    # 3. Remove non-alphabetical characters (keep letters and spaces)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    
    # 4. Normalize multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def categorize_location(loc_string: str) -> str:
    """
    Normalizes chaotic geographic metadata into structured geopolitical categories:
    - Remote Work
    - United States (Regional & Major Hubs)
    - Europe / International Hubs (UK, Germany, Canada, India, etc.)
    - Other

    Args:
        loc_string (str): Raw location string from job posting.

    Returns:
        str: Standardized location category.
    """
    if not isinstance(loc_string, str) or not loc_string.strip():
        return "Other"
    
    loc_lower = loc_string.lower().strip()
    
    # 1. Detect Remote / Hybrid
    if "remote" in loc_lower or "anywhere" in loc_lower or "work from home" in loc_lower:
        return "Remote"
    
    # 2. Detect United States & State Abbreviations
    us_states = [
        "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id", "il", "in", "ia",
        "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj",
        "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt",
        "va", "wa", "wv", "wi", "wy", "dc", "usa", "united states", "us"
    ]
    if "united states" in loc_lower or "usa" in loc_lower or "u.s." in loc_lower:
        return "United States"
    
    parts = [p.strip() for p in loc_lower.replace(",", " ").split()]
    for p in parts:
        if p in us_states:
            return "United States"
            
    # 3. Detect Major Global Tech Hubs
    if any(country in loc_lower for country in ["united kingdom", "uk", "london", "england", "scotland"]):
        return "United Kingdom"
    if any(country in loc_lower for country in ["germany", "berlin", "munich", "frankfurt"]):
        return "Germany"
    if any(country in loc_lower for country in ["canada", "toronto", "vancouver", "montreal"]):
        return "Canada"
    if any(country in loc_lower for country in ["india", "bangalore", "bengaluru", "hyderabad", "mumbai"]):
        return "India"
    if any(country in loc_lower for country in ["france", "paris"]):
        return "France"
        
    return "Other / International"


def parse_salary_range(salary_str: str) -> Optional[float]:
    """
    Extracts the numeric midpoint from salary strings (e.g. '$120,000 - $180,000 / yr').

    Args:
        salary_str (str): Raw salary text string.

    Returns:
        Optional[float]: Median annualized salary in USD, or None if unparseable.
    """
    if not isinstance(salary_str, str) or not salary_str.strip():
        return None
        
    clean_str = salary_str.replace("$", "").replace(",", "").lower()
    numbers = [float(s) for s in re.findall(r"\d+\.?\d*", clean_str)]
    
    if not numbers:
        return None
    if len(numbers) == 1:
        return numbers[0]
    return sum(numbers[:2]) / 2.0
