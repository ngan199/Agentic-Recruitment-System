from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from config.settings import NER_MODEL_NAME

def init_ner_pipeline(device: int = -1):
    """Create HR pipeline using aggregation strategy 'simple' and device setting."""
    tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME)
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, device=device, aggregation_strategy="simple")
    return ner_pipeline 

def normalize_ner(ner_results: list[dict]) -> list[dict]:
    """Normalize NER results to a consistent format."""
    normalized_results = []
    for entity in ner_results:
        normalized_entity = {
            "entity": entity.get("entity_group", ""),
            "score": entity.get("score", 0.0),
            "start": entity.get("start", 0),
            "end": entity.get("end", 0),
            "word": entity.get("word", "")
        }
        normalized_results.append(normalized_entity)

    return normalized_results


def run_ner(text: str) -> list[dict]:
    """Run NER pipeline and return a structured list of entities."""
    nlp = init_ner_pipeline()
    ner_results = nlp(text)
    normalized_results = normalize_ner(ner_results)
    return normalized_results
