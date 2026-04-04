import numpy as np

from arsl.preprocessing import construct_graph, get_adjacency_matrix, interpolate_frames, normalize_landmarks


def test_interpolate_frames():
    # Create 10 dummy frames (each 10x10x3)
    frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(10)]
    t = 20
    interpolated = interpolate_frames(frames, t)
    assert len(interpolated) == t
    assert interpolated[0].shape == (10, 10, 3)

    t = 5
    interpolated = interpolate_frames(frames, t)
    assert len(interpolated) == t


def test_normalize_landmarks():
    # 2 frames, each with 25 landmarks (x,y,z)
    # Frame 0: All zero (should remain zero)
    # Frame 1: Nose at (0.5, 0.5, 0.5), other landmarks shifted
    landmarks = np.zeros((2, 25, 3))
    landmarks[1, 0] = [0.5, 0.5, 0.5]
    landmarks[1, 1] = [1.0, 1.0, 1.0]

    normalized = normalize_landmarks(landmarks)

    assert normalized.shape == (2, 25, 3)
    assert np.all(normalized[0] == 0)
    # Nose should be at (0,0,0) after centering
    assert np.allclose(normalized[1, 0], [0, 0, 0])
    # Check if some normalization happened
    assert np.max(np.abs(normalized[1])) <= 1.0


def test_construct_graph():
    # 2 frames, 25 landmarks
    t = 2
    num_keypoints = 25
    landmarks = np.random.rand(t, num_keypoints, 3)
    g, x = construct_graph(landmarks)

    assert x.shape == (t * num_keypoints, 3)
    assert g.number_of_nodes() == t * num_keypoints

    # Spatial edges per frame: 4 (body) + 9 (left) + 9 (right) = 22
    # For 2 frames, total spatial edges should be 44
    assert g.number_of_edges() == 44

    # Check some specific edges
    # Frame 0: nose (0) connected to left shoulder (1)
    assert g.has_edge(0, 1)
    # Frame 1: nose (25) connected to left shoulder (26)
    assert g.has_edge(25, 26)
    # No connection between frame 0 and frame 1 (temporal edges not implemented in construct_graph)
    assert not g.has_edge(0, 25)


def test_get_adjacency_matrix():
    import networkx as nx

    g = nx.Graph()
    g.add_edge(0, 1)
    adj = get_adjacency_matrix(g)
    assert adj.shape == (2, 2)
    assert adj[0, 1] == 1
    assert adj[1, 0] == 1
