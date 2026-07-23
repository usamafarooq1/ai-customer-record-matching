import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from src.matching.fuzzy_matcher import FastBlockedMatcher

app = FastAPI(
    title="AI Entity Resolution & Customer Record Matching API",
    description="Enterprise API combining RapidFuzz string distance and Sentence Transformers vector embeddings.",
    version="1.0.0"
)

# Initialize engines once at startup
fuzzy_engine = FastBlockedMatcher()
print("Loading AI Embedding Model for API...")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')


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
        "docs_url": "/docs"
    }


@app.post("/match")
def match_records(request: MatchRequest):
    """Calculates hybrid deterministic + semantic match score for two customer records."""
    try:
        crm_dict = request.crm_record.dict()
        pub_dict = request.public_record.dict()

        # Add temporary IDs for matcher compatibility
        crm_dict["account_id"] = "CRM-TEMP"
        pub_dict["public_record_id"] = "PUB-TEMP"

        crm_series = pd.Series(crm_dict)
        pub_series = pd.Series(pub_dict)

        # 1. Fuzzy String Match Score (Fixed method name!)
        fuzzy_result = fuzzy_engine.calculate_pair_score(crm_series, pub_series)

        # 2. Semantic Vector Match Score
        crm_str = f"{crm_dict['first_name']} {crm_dict['last_name']}, {crm_dict['address']}, {crm_dict['city']} {crm_dict['postcode']}"
        pub_str = f"{pub_dict['first_name']} {pub_dict['last_name']}, {pub_dict['address']}, {pub_dict['city']} {pub_dict['postcode']}"

        emb_crm = semantic_model.encode(crm_str, convert_to_tensor=True)
        emb_pub = semantic_model.encode(pub_str, convert_to_tensor=True)

        cosine_score = float(util.cos_sim(emb_crm, emb_pub).numpy()[0][0])
        semantic_score = round(cosine_score * 100, 2)

        # 3. Hybrid Confidence Calculation (50% Fuzzy + 50% Semantic)
        hybrid_score = round((fuzzy_result["confidence_score"] * 0.5) + (semantic_score * 0.5), 2)

        return {
            "hybrid_confidence_score": hybrid_score,
            "is_match": hybrid_score >= 75.0,
            "breakdown": {
                "fuzzy_string_score": fuzzy_result["confidence_score"],
                "semantic_vector_score": semantic_score,
                "name_score": fuzzy_result["name_score"],
                "address_score": fuzzy_result["address_score"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))