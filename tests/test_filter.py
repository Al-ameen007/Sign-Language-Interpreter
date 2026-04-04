import os

import pandas as pd

from arsl.filter import filter_by_signid


def test_filter_by_signid(tmp_path):
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "output.csv"

    df = pd.DataFrame(
        {"file_path": ["p1", "p2", "p3", "p4"], "Label": ["l1", "l2", "l3", "l4"], "SignID": [1, 5, 10, 15]}
    )
    df.to_csv(input_csv, index=False)

    max_signid = 10
    filtered_df = filter_by_signid(input_csv, output_csv, max_signid)

    assert len(filtered_df) == 3
    assert filtered_df["SignID"].max() == 10
    assert os.path.exists(output_csv)

    # Read back from file
    df_read = pd.read_csv(output_csv)
    assert len(df_read) == 3
