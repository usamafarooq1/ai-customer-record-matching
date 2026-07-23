import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.matching.fuzzy_matcher import FuzzyMatcher
from src.matching.semantic_matcher import SemanticMatcher

app = FastAPI(
    title="AI Entity Resolution & Customer Record Matching API",
    description="Enterprise API combining RapidFuzz string distance and Sentence Transformers vector embeddings.",
    version="1.0.0",
)

# Initialize matching engines once on startup
fuzzy_engine = FuzzyMatcher()
semantic_engine = SemanticMatcher()


class CustomerRecord(BaseModel):
    first_name: str
    last_name: str
    address: str
    city: str
    postcode: str


class MatchRequest(BaseModel):
    crm_record: CustomerRecord
    public_record: CustomerRecord


@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "AI Customer Record Linkage Engine",
        "docs_url": "/docs",
    }


@app.post("/match")
def match_records(request: MatchRequest):
    """Calculates hybrid deterministic + semantic match score for two customer records."""
    crm_dict = request.crm_record.dict()
    pub_dict = request.public_record.dict()

    # Add dummy IDs for model compatibility
    crm_dict["account_id"] = "CRM-TEMP"
    pub_dict["public_record_id"] = "PUB-TEMP"

    crm_series = pd.Series(crm_dict)
    pub_series = pd.Series(pub_dict)

    # 1. Deterministic Fuzzy Match Score
    fuzzy_result = fuzzy_engine.calculate_similarity(crm_series, pub_series)

    # 2. Semantic Vector Match Score
    crm_str = semantic_engine.create_profile_string(crm_series)
    pub_str = semantic_engine.create_profile_string(pub_series)

    emb_crm = semantic_engine.model.encode(crm_str, convert_to_tensor=True)
    emb_pub = semantic_engine.model.encode(pub_str, convert_to_tensor=True)

    from sentence_transformers import util

    cosine_score = float(util.cos_sim(emb_crm, emb_pub).numpy()[0][0])
    semantic_score = round(cosine_score * 100, 2)

    # 3. Hybrid Confidence Calculation (50% Fuzzy + 50% Semantic)
    hybrid_score = round(
        (fuzzy_result["confidence_score"] * 0.5) + (semantic_score * 0.5), 2
    )

    return {
        "hybrid_confidence_score": hybrid_score,
        "is_match": hybrid_score >= 75.0,
        "breakdown": {
            "fuzzy_string_score": fuzzy_result["confidence_score"],
            "semantic_vector_score": semantic_score,
            "name_score": fuzzy_result["name_score"],
            "address_score": fuzzy_result["address_score"],
        },
    }