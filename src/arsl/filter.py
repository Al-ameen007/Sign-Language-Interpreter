import argparse

import pandas as pd


def filter_by_signid(input_csv, output_csv, max_signid):
    df = pd.read_csv(input_csv)

    print(f"Original: {len(df)} videos, {df['SignID'].nunique()} classes")

    filtered_df = df[df["SignID"] <= max_signid]

    print(f"Filtered: {len(filtered_df)} videos, {filtered_df['SignID'].nunique()} classes")
    print(f"SignID range: {filtered_df['SignID'].min()} to {filtered_df['SignID'].max()}")

    filtered_df.to_csv(output_csv, index=False)
    print(f"Saved to: {output_csv}")

    return filtered_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter CSV by SignID")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max-signid", type=int, required=True)

    args = parser.parse_args()

    filter_by_signid(args.input, args.output, args.max_signid)
