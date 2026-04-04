from unittest.mock import MagicMock, patch

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from arsl.inference import predict_from_csv, predict_video


def test_predict_video():
    # Mocking preprocessing functions
    with (
        patch("arsl.inference.sample_frames") as mock_sample,
        patch("arsl.inference.detect_landmarks_mediapipe") as mock_detect,
        patch("arsl.inference.normalize_landmarks") as mock_normalize,
        patch("arsl.inference.construct_graph") as mock_graph,
        patch("arsl.inference.get_adjacency_matrix") as mock_adj,
    ):
        mock_sample.return_value = [None] * 48
        mock_detect.return_value = MagicMock()
        mock_normalize.return_value = MagicMock()
        mock_graph.return_value = (MagicMock(), MagicMock())
        mock_adj.return_value = MagicMock()

        # Mock model and label encoder
        model = MagicMock()
        # Mock model forward pass returning a tensor with high score for class 1
        model.return_value = torch.tensor([[0.1, 0.9]])

        label_encoder = LabelEncoder()
        label_encoder.fit(["label0", "label1"])

        # We need to mock torch.tensor calls inside predict_video too,
        # or make sure the return values from mock_adj are convertible to tensor
        mock_adj.return_value = [[0, 1], [1, 0]]
        mock_graph.return_value = (MagicMock(), [[0.1, 0.1, 0.1]] * 1200)  # 48 * 25 nodes

        result = predict_video("dummy_path.mp4", model, label_encoder)

        assert result == "label1"


def test_predict_from_csv(tmp_path):
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({"file_path": ["v1.mp4", "v2.mp4"], "Label": ["l1", "l2"]})
    df.to_csv(csv_path, index=False)

    # Mock predict_video
    with patch("arsl.inference.predict_video") as mock_predict:
        mock_predict.side_effect = ["l1", "l0"]

        model = MagicMock()
        label_encoder = MagicMock()

        results_df = predict_from_csv(csv_path, model, label_encoder)

        assert len(results_df) == 2
        assert results_df.iloc[0]["predicted_label"] == "l1"
        assert results_df.iloc[0]["correct"]
        assert results_df.iloc[1]["predicted_label"] == "l0"
        assert not results_df.iloc[1]["correct"]
