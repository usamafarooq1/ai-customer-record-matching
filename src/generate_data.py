import json
import os
import random
import pandas as pd

# Set seed for reproducible dirty data generation
random.seed(42)

# Raw clean customer profiles
BASE_CUSTOMERS = [
    {
        "account_id": "CRM-1001",
        "first_name": "Usama",
        "last_name": "Farooq",
        "dob": "1996-05-14",
        "address": "124 High Street",
        "city": "Birmingham",
        "postcode": "B1 1AA",
        "debt_amount": 450.50,
    },
    {
        "account_id": "CRM-1002",
        "first_name": "Mubashra",
        "last_name": "Razzaq",
        "dob": "1998-11-20",
        "address": "45 Park Lane",
        "city": "London",
        "postcode": "W1K 1AA",
        "debt_amount": 1200.00,
    },
    {
        "account_id": "CRM-1003",
        "first_name": "Alexander",
        "last_name": "Smith",
        "dob": "1985-03-30",
        "address": "89 Station Road",
        "city": "Nottingham",
        "postcode": "NG1 2AB",
        "debt_amount": 85.20,
    },
    {
        "account_id": "CRM-1004",
        "first_name": "Mohammed",
        "last_name": "Al-Mansoor",
        "dob": "1991-08-05",
        "address": "12 Victoria Street",
        "city": "Manchester",
        "postcode": "M1 1AE",
        "debt_amount": 920.75,
    },
]


def introduce_dirty_variations(record: dict) -> dict:
    """Intentionally creates messy external public record variations."""
    dirty = record.copy()

    # Vary account ID prefix to simulate Electoral Roll / Debt Registry
    dirty["public_record_id"] = record["account_id"].replace("CRM", "PUB")

    # Name variations (middle initials, truncations)
    name_draw = random.random()
    if name_draw < 0.33:
        dirty["first_name"] = f"{record['first_name'][0]}."  # e.g., "U."
    elif name_draw < 0.66:
        dirty["first_name"] = f"{record['first_name']} M."  # e.g., "Usama M."

    # Address variations
    dirty["address"] = (
        dirty["address"].replace("Street", "St").replace("Road", "Rd")
    )

    # Postcode variation (removing spaces)
    if random.random() > 0.5:
        dirty["postcode"] = dirty["postcode"].replace(" ", "")

    # Drop fields not present in public registry
    del dirty["debt_amount"]
    del dirty["account_id"]

    return dirty


def generate_datasets():
    os.makedirs("data/raw", exist_ok=True)

    crm_df = pd.DataFrame(BASE_CUSTOMERS)
    public_records = [
        introduce_dirty_variations(rec) for rec in BASE_CUSTOMERS
    ]
    public_df = pd.DataFrame(public_records)

    crm_df.to_csv("data/raw/crm_records.csv", index=False)
    public_df.to_csv("data/raw/public_electoral_records.csv", index=False)

    print("Successfully generated CRM and Electoral public datasets!")
    print(f"CRM Records saved: {len(crm_df)}")
    print(f"Public Records saved: {len(public_df)}")


if __name__ == "__main__":
    generate_datasets()