from typing import Dict, Any
from .ner_helpers import run_ner

def entity_extractor(raw_text: str, *, do_zero_shot: bool = True) -> Dict[str, Any]:
    """
    High-level orchestrator that converts raw_text -> cv_dict.

    Args:
        raw_text: combined text from text_extractor 
        do_zero_shot: whether to run domain/seniority classification 
    
    Returns: 
        cv_dict with keys: name, email, phone, skills, education, experience, summary, job_domain, seniority
    """
    # Run NER 
    ner_output = run_ner(raw_text)      # from ner_helpers 

    # Extract regex-based structured fields 
    contacts = extract_contacts(raw_text)   # from regex_helpers# from regex_helpers 

    # Find skill candidates via embedding similarity 
    skill_candidates - match_skills(raw_text)   # from skill_matcher    # from skill_matcher

    # Pattern match titles & degrees 
    titles_degrees = extract_titles_degrees(raw_text)   # from regex_helpers or pattern module 

    # Group & merge entities -> entity_groups 
    entity_groups = group_entities(raw_text, ner_output, contacts, skill_candidates, titles_degress)

    # Optional zero-shot classification for domain/seniority
    if do_zero_shot:
        tags = zero_shot_classify(raw_text)   # from zero_shot_helpers
    else:
        tags = {}
    
    # Normalize & ma to final schema
    cv_dict = normalize_and_map(entity_groups, tags)

    return cv_dict 