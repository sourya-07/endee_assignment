import json
import os
from .config import SUBJECTS_DB

def load_subjects():
    if not os.path.exists(SUBJECTS_DB):
        return {}
    try:
        with open(SUBJECTS_DB, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_subjects(subjects):
    with open(SUBJECTS_DB, "w") as f:
        json.dump(subjects, f, indent=4)

def get_subject_index_name(subject_name):
    subjects = load_subjects()
    return subjects.get(subject_name, {}).get("index_name")

def create_or_get_subject(subject_name):
    subjects = load_subjects()
    if subject_name not in subjects:
        # Create a URL friendly index name
        index_name = "".join(c if c.isalnum() else "_" for c in subject_name).lower()
        if not index_name:
            index_name = "default_" + os.urandom(4).hex()
            
        subjects[subject_name] = {
            "index_name": index_name,
            "docs_count": 0,
            "created_at": __import__('datetime').datetime.now().isoformat()
        }
        save_subjects(subjects)
    
    return subjects[subject_name]["index_name"]
