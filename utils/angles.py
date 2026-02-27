import math
import torch
import torch.optim as optim
import numpy as np
from scipy.ndimage import gaussian_filter1d

AOI = [
    [12, 14, 18],
    [12, 14, 16],
    [11, 13, 15],
    [11, 13, 17],
    [10, 12, 14],
    [9, 11, 13],
    [4, 10, 12],
    [3, 9, 11],
    [4, 6, 10],
    [3, 5, 9],
    [11, 12, 14],
    [11, 12, 13],
]


def angle_between_lines(tensor_a, tensor_b, tensor_c, degrees=False):
    vector1 = tensor_b - tensor_a
    vector2 = tensor_c - tensor_b
    vector1_norm = vector1 / torch.norm(vector1)
    vector2_norm = vector2 / torch.norm(vector2)
    dot_product = torch.dot(vector1_norm, vector2_norm)
    dot_product = (
        torch.tensor(1)
        if dot_product > 1
        else torch.tensor(-1) if dot_product < -1 else dot_product
    )
    angle = torch.acos(dot_product)

    return math.degrees(angle) if degrees else angle


def compute_angles(points) -> torch.Tensor:
    angles = []
    for frame in points:
        frame_angles = []
        for joints in AOI:
            angle = angle_between_lines(
                frame[joints[0]], frame[joints[1]], frame[joints[2]]
            )
            frame_angles.append(angle)
        angles.append(torch.stack(frame_angles))
    return torch.stack(angles)


def ik_transform(original_smooth, data, c):
    data = data.reshape(-1, 19, 3)  # Replace with your actual data
    original_points = original_smooth.clone().detach().requires_grad_(True)
    target_angles = c  # Shape: (71, 12)

    optimizer = optim.Adam([original_points], lr=0.01)  # Shape: (71, 19, 3)
    num_iterations = 10

    for iteration in range(num_iterations):
        optimizer.zero_grad()

        current_angles = compute_angles(original_points).to(target_angles.device)
        loss = torch.mean(torch.abs(current_angles - target_angles))

        loss.backward()

        print(
            f"Iteration {iteration}: loss = {loss.item()} Gradients: {original_points.grad.norm()}"
        )

        optimizer.step()

    adjusted_points = original_points.detach()
    print("Adjusted points shape:", adjusted_points.shape)
    return adjusted_points


def plot_points(real_data, adj_data, joint=1):
    X_adj, Y_adj, Z_adj = [], [], []
    X_real, Y_real, Z_real = [], [], []
    for frame1, frame2 in zip(real_data, adj_data):
        x1, y1, z1 = frame1[joint].cpu()
        X_real.append(x1.item())
        Y_real.append(y1.item())
        Z_real.append(z1.item())
        x2, y2, z2 = frame2[joint].cpu()
        X_adj.append(x2.item())
        Y_adj.append(y2.item())
        Z_adj.append(z2.item())
    return X_real, Y_real, Z_real, X_adj, Y_adj, Z_adj


def gaussian_smooth(data, sigma=1):
    numpy_data = data.cpu().numpy()
    smoothed_data = np.zeros_like(numpy_data)
    for joint in range(19):
        for coord in range(3):
            smoothed_data[:, joint, coord] = gaussian_filter1d(
                numpy_data[:, joint, coord], sigma
            )
    return torch.from_numpy(smoothed_data)
