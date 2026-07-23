import pandas as pd
from rapidfuzz import fuzz, process


class FuzzyMatcher:

    def __init__(self, name_threshold: float = 70.0, overall_threshold: float = 75.0):
        self.name_threshold = name_threshold
        self.overall_threshold = overall_threshold

    def calculate_similarity(self, row_crm: pd.Series, row_pub: pd.Series) -> dict:
        """Calculates fuzzy similarity scores across name, address, and postcode."""
        # Full name comparison
        crm_name = f"{row_crm['first_name']} {row_crm['last_name']}".lower()
        pub_name = f"{row_pub['first_name']} {row_pub['last_name']}".lower()

        # Token sort ratio handles middle initials and name ordering
        name_score = fuzz.token_sort_ratio(crm_name, pub_name)

        # Address fuzzy matching
        addr_score = fuzz.token_set_ratio(
            str(row_crm['address']).lower(),
            str(row_pub['address']).lower()
        )

        # Postcode matching (exact match after stripping spaces)
        crm_pc = str(row_crm['postcode']).replace(" ", "").lower()
        pub_pc = str(row_pub['postcode']).replace(" ", "").lower()
        postcode_score = 100.0 if crm_pc == pub_pc else 0.0

        # Weighted composite confidence score
        overall_score = (name_score * 0.4) + (addr_score * 0.4) + (postcode_score * 0.2)

        return {
            "account_id": row_crm["account_id"],
            "public_record_id": row_pub["public_record_id"],
            "name_score": round(name_score, 2),
            "address_score": round(addr_score, 2),
            "postcode_score": round(postcode_score, 2),
            "confidence_score": round(overall_score, 2),
            "is_match": overall_score >= self.overall_threshold
        }

    def match_datasets(self, crm_df: pd.DataFrame, public_df: pd.DataFrame) -> pd.DataFrame:
        """Compares all CRM records against Public records to find potential matches."""
        results = []
        for _, crm_row in crm_df.iterrows():
            for _, pub_row in public_df.iterrows():
                res = self.calculate_similarity(crm_row, pub_row)
                if res["is_match"]:
                    results.append(res)

        return pd.DataFrame(results)


if __name__ == "__main__":
    crm_df = pd.read_csv("data/raw/crm_records.csv")
    pub_df = pd.read_csv("data/raw/public_electoral_records.csv")

    matcher = FuzzyMatcher()
    matches_df = matcher.match_datasets(crm_df, pub_df)

    print("\n--- Deterministic Fuzzy Match Results ---")
    print(matches_df[["account_id", "public_record_id", "confidence_score", "is_match"]])