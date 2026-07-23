import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util


class SemanticMatcher:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Load lightweight MiniLM embedding model
        print(f"Loading AI Embedding Model ({model_name})...")
        self.model = SentenceTransformer(model_name)

    def create_profile_string(self, row: pd.Series) -> str:
        """Combines entity attributes into a single semantic profile text."""
        return f"Name: {row['first_name']} {row['last_name']} | Address: {row['address']}, {row['city']} | Postcode: {row['postcode']}"

    def match_datasets(
        self,
        crm_df: pd.DataFrame,
        public_df: pd.DataFrame,
        similarity_threshold: float = 0.70,
    ) -> pd.DataFrame:
        """Generates vector embeddings and computes cosine similarity matrix."""
        # Build text profiles for vector representation
        crm_profiles = crm_df.apply(self.create_profile_string, axis=1).tolist()
        public_profiles = public_df.apply(
            self.create_profile_string, axis=1
        ).tolist()

        # Generate vector embeddings
        crm_embeddings = self.model.encode(crm_profiles, convert_to_tensor=True)
        public_embeddings = self.model.encode(
            public_profiles, convert_to_tensor=True
        )

        # Compute cosine similarity matrix
        cosine_scores = util.cos_sim(crm_embeddings, public_embeddings).numpy()

        results = []
        for idx_crm, row_crm in crm_df.iterrows():
            for idx_pub, row_pub in public_df.iterrows():
                score = float(cosine_scores[idx_crm][idx_pub])
                if score >= similarity_threshold:
                    results.append(
                        {
                            "account_id": row_crm["account_id"],
                            "public_record_id": row_pub["public_record_id"],
                            "ai_vector_score": round(score * 100, 2),
                            "is_semantic_match": True,
                        }
                    )

        return pd.DataFrame(results)


if __name__ == "__main__":
    crm_df = pd.read_csv("data/raw/crm_records.csv")
    pub_df = pd.read_csv("data/raw/public_electoral_records.csv")

    matcher = SemanticMatcher()
    ai_matches_df = matcher.match_datasets(crm_df, pub_df)

    print("\n--- AI Vector Semantic Match Results ---")
    print(ai_matches_df)