import cv2
import mediapipe as mp
import numpy as np
import networkx as nx
from scipy.interpolate import interp1d
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def sample_frames(video_path, T=48):
    cap = cv2.VideoCapture(video_path)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    if N < T:
        for i in range(N):
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        frames = interpolate_frames(frames, T)
    else:
        I = max(1, N // T)
        for i in range(0, N, I):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        if len(frames) < T:
            frames = interpolate_frames(frames, T)
        elif len(frames) > T:
            frames = frames[:T]
    
    cap.release()
    return frames


def interpolate_frames(frames, T):
    N = len(frames)
    if N == 0:
        return []
    
    indices = np.arange(N)
    new_indices = np.linspace(0, N - 1, T)
    
    interpolated_frames = []
    for new_idx in new_indices:
        lower_idx = int(np.floor(new_idx))
        upper_idx = int(np.ceil(new_idx))
        
        if lower_idx == upper_idx:
            interpolated_frames.append(frames[lower_idx])
        else:
            alpha = new_idx - lower_idx
            lower_frame = frames[lower_idx].astype(np.float32)
            upper_frame = frames[upper_idx].astype(np.float32)
            interpolated_frame = (1 - alpha) * lower_frame + alpha * upper_frame
            interpolated_frames.append(interpolated_frame.astype(np.uint8))
    
    return interpolated_frames


def detect_landmarks_mediapipe(frames):
    mp_holistic = mp.solutions.holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5
    )
    
    selected_hand_indices = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]
    
    all_landmarks = []
    
    for frame in frames:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_holistic.process(rgb_frame)
        
        frame_landmarks = []
        
        if results.pose_landmarks:
            nose = results.pose_landmarks.landmark[0]
            left_shoulder = results.pose_landmarks.landmark[11]
            right_shoulder = results.pose_landmarks.landmark[12]
            left_elbow = results.pose_landmarks.landmark[13]
            right_elbow = results.pose_landmarks.landmark[14]
            
            frame_landmarks.append([nose.x, nose.y, nose.z])
            frame_landmarks.append([left_shoulder.x, left_shoulder.y, left_shoulder.z])
            frame_landmarks.append([right_shoulder.x, right_shoulder.y, right_shoulder.z])
            frame_landmarks.append([left_elbow.x, left_elbow.y, left_elbow.z])
            frame_landmarks.append([right_elbow.x, right_elbow.y, right_elbow.z])
        else:
            frame_landmarks.extend([[0.0, 0.0, 0.0] for _ in range(5)])
        
        if results.left_hand_landmarks:
            for idx in selected_hand_indices:
                lm = results.left_hand_landmarks.landmark[idx]
                frame_landmarks.append([lm.x, lm.y, lm.z])
        else:
            frame_landmarks.extend([[0.0, 0.0, 0.0] for _ in range(10)])
        
        if results.right_hand_landmarks:
            for idx in selected_hand_indices:
                lm = results.right_hand_landmarks.landmark[idx]
                frame_landmarks.append([lm.x, lm.y, lm.z])
        else:
            frame_landmarks.extend([[0.0, 0.0, 0.0] for _ in range(10)])
        
        all_landmarks.append(frame_landmarks)
    
    mp_holistic.close()
    return np.array(all_landmarks)


def normalize_landmarks(landmarks):
    normalized = []
    
    for frame_landmarks in landmarks:
        frame_landmarks = np.array(frame_landmarks)
        
        if np.all(frame_landmarks == 0):
            normalized.append(frame_landmarks)
            continue
        
        nose = frame_landmarks[0]
        centered = frame_landmarks - nose
        
        distances = np.linalg.norm(centered, axis=1)
        max_distance = np.max(distances)
        
        if max_distance > 0:
            normalized_frame = centered / max_distance
        else:
            normalized_frame = centered
        
        normalized.append(normalized_frame)
    
    return np.array(normalized)


def construct_graph(landmarks):

    T = landmarks.shape[0]  
    num_keypoints = 25

    body_connections = [
        (0, 1), (0, 2), (1, 3), (2, 4)
    ]

    left_hand_connections = [
        (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11),
        (5, 12), (12, 13), (13, 14)
    ]

    right_hand_connections = [
        (15, 16), (16, 17), (17, 18),
        (15, 19), (19, 20), (20, 21),
        (15, 22), (22, 23), (23, 24)
    ]

    spatial_edges = body_connections + left_hand_connections + right_hand_connections

    G = nx.Graph()

    # Add all nodes (T frames * 25 keypoints)
    total_nodes = T * num_keypoints
    for i in range(total_nodes):
        G.add_node(i)

    # Add spatial edges for each frame
    for t in range(T):
        offset = t * num_keypoints
        for (i, j) in spatial_edges:
            G.add_edge(offset + i, offset + j)

    flattened_landmarks = landmarks.reshape(-1, 3)
    X = flattened_landmarks

    return G, X


def get_adjacency_matrix(graph):
    return nx.adjacency_matrix(graph).todense()


def process_video(video_path, T=48):
    frames = sample_frames(video_path, T)
    landmarks = detect_landmarks_mediapipe(frames)
    normalized_landmarks = normalize_landmarks(landmarks)
    graph, X = construct_graph(normalized_landmarks)
    adj_matrix = get_adjacency_matrix(graph)
    
    return X, adj_matrix


def process_videos_parallel(file_paths, labels, max_workers=4):
    data_list = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_video, path): (path, label) 
                  for path, label in zip(file_paths, labels)}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
            path, label = futures[future]
            try:
                X, adj_matrix = future.result()
                data_list.append((X, adj_matrix, label))
            except Exception as e:
                print(f"Error processing {path}: {e}")
    
    return data_list