import os

import pandas as pd

from arsl.data_preparation import (
    assign_labels,
    copy_contents,
    create_combined_folder,
    remove_leading_zeros_from_folders,
)


def test_create_combined_folder(tmp_path):
    base_path = tmp_path
    folder_name = "test_combined"
    combined_path = create_combined_folder(base_path, folder_name)
    assert os.path.exists(combined_path)
    assert os.path.basename(combined_path) == folder_name


def test_copy_contents(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    (src / "file.txt").write_text("hello")
    (src / "subdir").mkdir()
    (src / "subdir" / "other.txt").write_text("world")

    copy_contents(src, dst)

    assert (dst / "file.txt").read_text() == "hello"
    assert (dst / "subdir" / "other.txt").read_text() == "world"


def test_remove_leading_zeros_from_folders(tmp_path):
    dir_path = tmp_path / "data"
    dir_path.mkdir()
    (dir_path / "0001").mkdir()
    (dir_path / "0010").mkdir()
    (dir_path / "9999").mkdir()  # Not leading zero
    (dir_path / "001-070").mkdir()
    (dir_path / "001-070" / "0001").mkdir()

    # Note: the current implementation only renames if item.startswith("0") and len(item) == 4
    remove_leading_zeros_from_folders(dir_path)

    assert os.path.exists(dir_path / "1")
    assert os.path.exists(dir_path / "10")
    assert os.path.exists(dir_path / "9999")
    assert os.path.exists(dir_path / "001-070" / "1")


def test_assign_labels(tmp_path):
    files_path = tmp_path / "files"
    files_path.mkdir()
    (files_path / "1").mkdir()
    (files_path / "1" / "video1.mp4").write_text("v1")
    (files_path / "2").mkdir()
    (files_path / "2" / "video2.mp4").write_text("v2")

    df_labels = pd.DataFrame({"SignID": [1, 2], "Sign-Arabic": ["label1", "label2"]})

    data = assign_labels(files_path, df_labels)

    assert len(data) == 2
    assert set(data["Label"]) == {"label1", "label2"}
    assert set(data["SignID"]) == {1, 2}
