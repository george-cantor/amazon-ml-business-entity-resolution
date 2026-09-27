import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE2_PATH = PROJECT_ROOT / "data" / "train" / "train_source2.tsv"
SOURCE3_PATH = PROJECT_ROOT / "data" / "train" / "train_source3.tsv"


def analyze_source(path, source_name, chunksize=200_000):

    print("\n" + "=" * 70)
    print(f"{source_name} ANALYSIS")
    print("=" * 70)

    total_rows = 0
    countries = {}
    empty_counts = {}
    duplicate_ids = set()
    seen_ids = set()

    first_chunk = True

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        keep_default_na=False,
        chunksize=chunksize
    ):

        total_rows += len(chunk)

        # --------------------------------------------
        # Columns
        # --------------------------------------------

        if first_chunk:
            print("\nColumns:")
            print(chunk.columns.tolist())

            print("\nFirst 5 rows:")
            print(chunk.head().to_string(index=False))

            empty_counts = {
                col: 0
                for col in chunk.columns
            }

            first_chunk = False

        # --------------------------------------------
        # Empty values
        # --------------------------------------------

        for col in chunk.columns:
            empty_counts[col] += (
                chunk[col].str.strip() == ""
            ).sum()

        # --------------------------------------------
        # Country
        # --------------------------------------------

        country_counts = chunk["country"].value_counts()

        for country, count in country_counts.items():
            countries[country] = (
                countries.get(country, 0) + count
            )

        # --------------------------------------------
        # Duplicate IDs
        # --------------------------------------------

        ids = chunk["entity_id"].tolist()

        for entity_id in ids:

            if entity_id in seen_ids:
                duplicate_ids.add(entity_id)

            seen_ids.add(entity_id)

    print("\nTotal rows:")
    print(f"{total_rows:,}")

    print("\nUnique IDs:")
    print(f"{len(seen_ids):,}")

    print("\nDuplicate IDs:")
    print(f"{len(duplicate_ids):,}")

    print("\nCountry distribution:")

    for country, count in sorted(
        countries.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(
            f"{country:<15} "
            f"{count:>12,} "
            f"({count / total_rows * 100:>6.2f}%)"
        )

    print("\nEmpty values:")

    for col, count in empty_counts.items():

        print(
            f"{col:<20} "
            f"{count:>12,} "
            f"({count / total_rows * 100:>6.2f}%)"
        )


# ============================================================
# Run
# ============================================================

analyze_source(
    SOURCE2_PATH,
    "SOURCE 2"
)

analyze_source(
    SOURCE3_PATH,
    "SOURCE 3"
)