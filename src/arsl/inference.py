import cv2
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm

from arsl.preprocessing import (
    construct_graph,
    detect_landmarks_mediapipe,
    get_adjacency_matrix,
    normalize_landmarks,
    sample_frames,
)


def predict_video(video_path, model, label_encoder, device="cpu", t=48):
    frames = sample_frames(video_path, t)
    landmarks = detect_landmarks_mediapipe(frames)
    normalized_landmarks = normalize_landmarks(landmarks)
    graph, x = construct_graph(normalized_landmarks)
    adj_matrix = get_adjacency_matrix(graph)

    edge_index, _ = dense_to_sparse(torch.tensor(adj_matrix, dtype=torch.float))
    edge_index = edge_index.long()
    x = torch.tensor(x, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    data = data.to(device)

    model.eval()
    with torch.no_grad():
        output = model(data.x, data.edge_index, data.batch)
        predicted_class = output.argmax(dim=1).item()

    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

    return predicted_label


def predict_from_csv(csv_path, model, label_encoder, device="cpu", t=48):
    df = pd.read_csv(csv_path)

    if "file_path" not in df.columns:
        raise ValueError("CSV must have 'file_path' column")

    results = []

    print(f"Processing {len(df)} videos from CSV...")

    for _idx, row in tqdm(df.iterrows(), total=len(df)):
        video_path = row["file_path"]
        true_label = row.get("Label", None)

        try:
            predicted_label = predict_video(video_path, model, label_encoder, device, t)

            result = {
                "video_path": video_path,
                "predicted_label": predicted_label,
                "true_label": true_label,
                "correct": predicted_label == true_label if true_label else None,
            }
            results.append(result)

        except Exception as e:
            print(f"\nError processing {video_path}: {e}")
            results.append(
                {
                    "video_path": video_path,
                    "predicted_label": None,
                    "true_label": true_label,
                    "correct": None,
                    "error": str(e),
                }
            )

    results_df = pd.DataFrame(results)

    if "correct" in results_df.columns and results_df["correct"].notna().any():
        accuracy = results_df["correct"].mean()
        print(f"\nAccuracy: {accuracy:.4f} ({results_df['correct'].sum()}/{results_df['correct'].notna().sum()})")

    return results_df


def predict_live_webcam(model, label_encoder, device="cpu", buffer_size=48, display=True):
    from collections import deque

    cap = cv2.VideoCapture(0)
    frame_buffer = deque(maxlen=buffer_size)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("\n" + "=" * 60)
    print("WEBCAM INFERENCE MODE")
    print("=" * 60)
    print("Controls:")
    print("  [SPACE] - Predict current buffer")
    print("  [R]     - Reset buffer")
    print("  [Q]     - Quit")
    print("=" * 60)

    last_prediction = "No prediction yet"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame_buffer.append(frame.copy())

        if display:
            display_frame = frame.copy()

            buffer_status = f"Buffer: {len(frame_buffer)}/{buffer_size}"
            cv2.putText(display_frame, buffer_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(
                display_frame,
                f"Prediction: {last_prediction}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            status_color = (0, 255, 0) if len(frame_buffer) == buffer_size else (0, 165, 255)
            cv2.putText(
                display_frame,
                "READY" if len(frame_buffer) == buffer_size else "BUFFERING...",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2,
            )

            cv2.imshow("Sign Language Interpreter", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("\nQuitting...")
            break

        elif key == ord("r"):
            frame_buffer.clear()
            last_prediction = "Buffer reset"
            print("\nBuffer cleared")

        elif key == ord(" ") and len(frame_buffer) == buffer_size:
            print("\nPredicting...")
            try:
                frames = list(frame_buffer)
                landmarks = detect_landmarks_mediapipe(frames)
                normalized_landmarks = normalize_landmarks(landmarks)
                graph, x = construct_graph(normalized_landmarks)
                adj_matrix = get_adjacency_matrix(graph)

                edge_index, _ = dense_to_sparse(torch.tensor(adj_matrix, dtype=torch.float))
                edge_index = edge_index.long()
                x = torch.tensor(x, dtype=torch.float)

                data = Data(x=x, edge_index=edge_index)
                data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
                data = data.to(device)

                model.eval()
                with torch.no_grad():
                    output = model(data.x, data.edge_index, data.batch)
                    predicted_class = output.argmax(dim=1).item()
                    confidence = torch.softmax(output, dim=1).max().item()

                predicted_label = label_encoder.inverse_transform([predicted_class])[0]
                last_prediction = f"{predicted_label} ({confidence:.2f})"
                print(f"Predicted: {last_prediction}")

            except Exception as e:
                last_prediction = f"Error: {str(e)}"
                print(f"Prediction error: {e}")

    cap.release()
    cv2.destroyAllWindows()
    print("\nWebcam inference stopped")


def batch_predict_videos(video_paths, model, label_encoder, device="cpu", t=48):
    results = []

    print(f"Processing {len(video_paths)} videos...")

    for video_path in tqdm(video_paths):
        try:
            predicted_label = predict_video(video_path, model, label_encoder, device, t)
            results.append({"video_path": video_path, "predicted_label": predicted_label})
        except Exception as e:
            print(f"\nError processing {video_path}: {e}")
            results.append({"video_path": video_path, "predicted_label": None, "error": str(e)})

    return pd.DataFrame(results)
