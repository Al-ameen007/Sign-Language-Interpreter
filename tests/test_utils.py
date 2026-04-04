import pandas as pd
import torch

from arsl.utils import get_device, load_data_csv


def test_load_data_csv(tmp_path):
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"

    train_df = pd.DataFrame(
        {"file_path": ["path/to/video1.mp4", "path/to/video2.mp4"], "Label": ["label1", "label2"], "SignID": [1, 2]}
    )
    test_df = pd.DataFrame({"file_path": ["path/to/video3.mp4"], "Label": ["label1"], "SignID": [1]})

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    x_train, y_train, x_test, y_test = load_data_csv(train_csv, test_csv)

    assert len(x_train) == 2
    assert len(y_train) == 2
    assert len(x_test) == 1
    assert len(y_test) == 1
    assert x_train.iloc[0] == "path/to/video1.mp4"
    assert y_train.iloc[1] == "label2"


def test_get_device():
    device = get_device()
    assert isinstance(device, torch.device)
    if torch.cuda.is_available():
        assert device.type == "cuda"
    else:
        assert device.type == "cpu"
