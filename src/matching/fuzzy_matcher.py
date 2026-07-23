from typing import Dict, List
import pandas as pd
from rapidfuzz import fuzz


class FastBlockedMatcher:

    def __init__(
        self, name_threshold: float = 70.0, overall_threshold: float = 75.0
    ):
        self.name_threshold = name_threshold
        self.overall_threshold = overall_threshold

    def _extract_block_key(self, postcode: str) -> str:
        clean_pc = str(postcode).replace(" ", "").lower()
        return clean_pc[:4] if len(clean_pc) >= 4 else clean_pc

    def calculate_pair_score(
        self, row_crm: pd.Series, row_pub: pd.Series
    ) -> Dict:
        crm_name = f"{row_crm['first_name']} {row_crm['last_name']}".lower()
        pub_name = f"{row_pub['first_name']} {row_pub['last_name']}".lower()

        name_score = fuzz.token_sort_ratio(crm_name, pub_name)
        addr_score = fuzz.token_set_ratio(
            str(row_crm["address"]).lower(), str(row_pub["address"]).lower()
        )

        crm_pc = str(row_crm["postcode"]).replace(" ", "").lower()
        pub_pc = str(row_pub["postcode"]).replace(" ", "").lower()
        postcode_score = 100.0 if crm_pc == pub_pc else 0.0

        overall_score = (
            (name_score * 0.4) + (addr_score * 0.4) + (postcode_score * 0.2)
        )

        return {
            "account_id": row_crm["account_id"],
            "public_record_id": row_pub["public_record_id"],
            "name_score": round(name_score, 2),
            "address_score": round(addr_score, 2),
            "postcode_score": round(postcode_score, 2),
            "confidence_score": round(overall_score, 2),
            "is_match": overall_score >= self.overall_threshold,
        }

    def match_datasets_blocked(
        self, crm_df: pd.DataFrame, public_df: pd.DataFrame
    ) -> pd.DataFrame:
        public_blocks: Dict[str, List[pd.Series]] = {}
        for _, pub_row in public_df.iterrows():
            b_key = self._extract_block_key(pub_row["postcode"])
            public_blocks.setdefault(b_key, []).append(pub_row)

        results = []
        total_evaluations = 0

        for _, crm_row in crm_df.iterrows():
            crm_b_key = self._extract_block_key(crm_row["postcode"])
            candidates = public_blocks.get(crm_b_key, [])

            for pub_row in candidates:
                total_evaluations += 1
                score_dict = self.calculate_pair_score(crm_row, pub_row)
                if score_dict["is_match"]:
                    results.append(score_dict)

        print(
            f"⚡ Blocking Optimization: Evaluated {total_evaluations:,} candidate pairs instead of {len(crm_df) * len(public_df):,} full comparisons!"
        )
        return pd.DataFrame(results)


if __name__ == "__main__":
    crm_df = pd.read_csv("data/raw/crm_records.csv")
    pub_df = pd.read_csv("data/raw/public_electoral_records.csv")

    matcher = FastBlockedMatcher()
    matches = matcher.match_datasets_blocked(crm_df, pub_df)

    print(f"Total High-Confidence Matches Found: {len(matches)}")
    print(matches.head(10))