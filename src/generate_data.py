import os
import random
from faker import Faker
import pandas as pd

fake = Faker("en_GB")
random.seed(42)
Faker.seed(42)


def introduce_noise(record: dict) -> dict:
    """Simulates real-world data noise (OCR typos, abbreviations, missing parts)."""
    dirty = record.copy()
    dirty["public_record_id"] = record["account_id"].replace("CRM", "PUB")

    # Name noise
    rand_name = random.random()
    if rand_name < 0.2:
        dirty["first_name"] = dirty["first_name"][0]
    elif rand_name < 0.4:
        dirty["first_name"] = f"{dirty['first_name']} {fake.first_name()[0]}."

    # Address noise
    address = dirty["address"]
    replacements = {
        "Street": "St",
        "Road": "Rd",
        "Avenue": "Ave",
        "Drive": "Dr",
        "Lane": "Ln",
    }
    for k, v in replacements.items():
        if k in address and random.random() > 0.4:
            address = address.replace(k, v)
    dirty["address"] = address

    # Postcode noise
    if random.random() < 0.5:
        dirty["postcode"] = dirty["postcode"].replace(" ", "")

    del dirty["debt_amount"]
    del dirty["account_id"]
    return dirty


def generate_large_datasets(num_records: int = 1000):
    os.makedirs("data/raw", exist_ok=True)
    crm_data = []

    print(f"Generating {num_records} realistic UK customer records...")

    for i in range(1, num_records + 1):
        crm_data.append(
            {
                "account_id": f"CRM-{10000 + i}",
                "first_name": fake.first_name(),
                "last_name": fake.last_name(),
                "dob": str(
                    fake.date_of_birth(minimum_age=18, maximum_age=80)
                ),
                "address": fake.street_address(),
                "city": fake.city(),
                "postcode": fake.postcode(),
                "debt_amount": round(random.uniform(50.0, 5000.0), 2),
            }
        )

    crm_df = pd.DataFrame(crm_data)
    public_records = [introduce_noise(rec) for rec in crm_data]
    random.shuffle(public_records)
    public_df = pd.DataFrame(public_records)

    crm_df.to_csv("data/raw/crm_records.csv", index=False)
    public_df.to_csv("data/raw/public_electoral_records.csv", index=False)

    print(
        f"✅ Datasets generated successfully in 'data/raw/' ({num_records} records each)."
    )


if __name__ == "__main__":
    generate_large_datasets(1000)