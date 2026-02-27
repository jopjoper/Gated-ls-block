# ==============================================================================
# Copyright (c) 2024 - Present. Your Name or Organization.
# All rights reserved.
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Bird 3D Pose Lifter
====================

This module provides a class and script to lift 2D bird keypoints (from MMPose or other detectors)
into a 3D space using numerical optimization. It enforces bone length priors and left-right symmetry.

Key Features:
    - Optimization-based 3D reconstruction (Levenberg-Marquardt).
    - Built-in skeleton connectivity and bone ratio priors (specific to the AnimalKingdom dataset topology).
    - Visualization tools for both 2D input and 3D output.

"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares
import cv2
import torch

# ==============================================================================
# CONSTANTS & CONFIGURATION
# ==============================================================================

# Mapping of keypoint indices to semantic names for the 23-point bird skeleton.
KEYPOINT_NAMES = {
    0: 'Head_Mid_Top', 1: 'Eye_Left', 2: 'Eye_Right',
    3: 'Mouth_Front_Top', 4: 'Mouth_Back_Left', 5: 'Mouth_Back_Right', 6: 'Mouth_Front_Bottom',
    7: 'Shoulder_Left', 8: 'Shoulder_Right',
    9: 'Elbow_Left', 10: 'Elbow_Right',
    11: 'Wrist_Left', 12: 'Wrist_Right',
    13: 'Torso_Mid_Back',  # [ROOT] Root node for kinematic chain
    14: 'Hip_Left', 15: 'Hip_Right',
    16: 'Knee_Left', 17: 'Knee_Right',
    18: 'Ankle_Left', 19: 'Ankle_Right',
    20: 'Tail_Top_Back', 21: 'Tail_Mid_Back', 22: 'Tail_End_Back',
}

# Defines the parent-child relationships for the skeleton used in optimization.
OPTIMIZATION_SKELETON = [
    # --- Spine ---
    (13, 0), (13, 20), (20, 21), (21, 22),
    # --- Torso ---
    (13, 7), (13, 8), (13, 14), (13, 15),
    # --- Head ---
    (0, 1), (0, 2), (0, 3), (3, 6), (3, 4), (3, 5),
    # --- Wings ---
    (7, 9), (9, 11), (8, 10), (10, 12),
    # --- Legs ---
    (14, 16), (16, 18), (15, 17), (17, 19)
]

# Bone length priors defined as ratios relative to the baseline (Torso to Tail Root, 13->20 = 1.0).
BONE_RATIO_PRIORS = {
    (13, 0): 0.8, (13, 20): 1.0, (20, 21): 0.5, (21, 22): 0.5,
    (13, 7): 0.3, (13, 8): 0.3, (7, 9): 0.7, (9, 11): 0.8,
    (8, 10): 0.7, (10, 12): 0.8, (13, 14): 0.25, (13, 15): 0.25,
    (14, 16): 0.5, (16, 18): 0.4, (15, 17): 0.5, (17, 19): 0.4,
    (0, 1): 0.15, (0, 2): 0.15, (0, 3): 0.25
}

# Pairs of indices that should be symmetric with respect to the root (Torso).
SYMMETRY_PAIRS = [
    (7, 8), (9, 10), (11, 12), (14, 15),
    (16, 17), (18, 19), (1, 2), (4, 5)
]


# ==============================================================================
# CORE CLASS
# ==============================================================================

class BirdPoseLifter:
    """
    A class to manage the 3D lifting process for bird skeletons.

    This class encapsulates the optimization logic, including the objective function
    that combines reprojection error, bone length constraints, and symmetry.
    """

    def __init__(self):
        self.num_points = 23
        self.edges = OPTIMIZATION_SKELETON
        self.ratios = BONE_RATIO_PRIORS
        self.sym_pairs = SYMMETRY_PAIRS

    def optimize_3d(self, points_2d, image_w, image_h):
        """
        Reconstructs 3D keypoints from 2D detections.

        Args:
            points_2d (np.ndarray): Input array of shape (23, 2) containing 2D coordinates (x, y).
            image_w (int): Width of the original image in pixels.
            image_h (int): Height of the original image in pixels.

        Returns:
            np.ndarray: Reconstructed 3D points of shape (23, 3). The origin is centered at the torso (idx 13),
                        but translated back to image coordinates for visualization convenience.
        """
        # 1. Normalize 2D coordinates to the range [-1, 1] centered in the image
        scale_factor = max(image_w, image_h)
        norm_2d = (points_2d - np.array([image_w / 2, image_h / 2])) / scale_factor

        # 2. Dynamic scale estimation based on torso/head size in 2D
        trunk_vec = norm_2d[13] - norm_2d[20]
        trunk_len_2d = np.linalg.norm(trunk_vec)
        head_vec = norm_2d[13] - norm_2d[0]
        head_len_2d = np.linalg.norm(head_vec)
        current_scale = max(trunk_len_2d, head_len_2d, 0.05)

        def objective_func(vars):
            """Internal objective function for least_squares."""
            P = vars.reshape((self.num_points, 3))

            # Term 1: 2D Reprojection Error (XY plane)
            reproj_err = (P[:, :2] - norm_2d).flatten() * 20.0

            # Term 2: Bone Length Consistency
            bone_errs = []
            for (u, v) in self.edges:
                curr_len = np.linalg.norm(P[u] - P[v])
                if (u, v) in self.ratios:
                    target_len = self.ratios[(u, v)] * current_scale
                    bone_errs.append((curr_len - target_len) * 2.0)
                else:
                    bone_errs.append((curr_len - 0.1 * current_scale) * 0.5)

            # Term 3: Symmetry Constraint (Distance from Root)
            sym_errs = []
            root_idx = 13
            for (left_idx, right_idx) in self.sym_pairs:
                dist_l = np.linalg.norm(P[left_idx] - P[root_idx])
                dist_r = np.linalg.norm(P[right_idx] - P[root_idx])
                sym_errs.append((dist_l - dist_r) * 5.0)

            # Term 4: Depth Regularization (Prevent Z explosion)
            z_reg = P[:, 2] * 0.5

            return np.concatenate([reproj_err, bone_errs, sym_errs, z_reg])

        # 3. Initialization
        init_z = np.zeros((self.num_points, 1))
        init_z[11] = 0.1  # Initial heuristic: Left wrist forward
        init_z[12] = -0.1  # Initial heuristic: Right wrist backward
        x0 = np.hstack([norm_2d, init_z]).flatten()

        # 4. Run Optimization
        res = least_squares(objective_func, x0, method='lm', max_nfev=100)

        # 5. Post-process results
        pts_3d_norm = res.x.reshape((self.num_points, 3))
        pts_3d = pts_3d_norm * scale_factor

        # Center the point cloud at the Torso (13)
        center = pts_3d[13].copy()
        pts_3d -= center

        # Translate back to image coordinate system for intuitive visualization
        pts_3d[:, 0] += image_w / 2
        pts_3d[:, 1] += image_h / 2

        return pts_3d


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def load_image(img_path):
    """Loads an image or creates a dummy black background if path is invalid."""
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            return img
    print(f"[Warning] Image not found at {img_path}. Generating black background for demo.")
    return np.zeros((800, 800, 3), dtype=np.uint8)


def generate_dummy_2d(w=800, h=800):
    """Generates a synthetic 23-point skeleton for testing purposes."""
    points_2d = np.zeros((23, 2)) + np.array([w / 2, h / 2])

    # Simple "Flying" pose configuration
    points_2d[13] = [w / 2, h / 2]  # Torso
    points_2d[20] = [w / 2 - 100, h / 2]  # Tail Root
    points_2d[0] = [w / 2 + 100, h / 2 - 20]  # Head

    # Left Wing
    points_2d[7] = [w / 2 + 20, h / 2 - 20]
    points_2d[9] = [w / 2 + 20, h / 2 - 100]
    points_2d[11] = [w / 2 + 50, h / 2 - 200]

    # Right Wing
    points_2d[8] = [w / 2 + 20, h / 2 + 20]
    points_2d[10] = [w / 2 + 20, h / 2 + 100]
    points_2d[12] = [w / 2 + 50, h / 2 + 200]

    # Add slight noise for realism
    points_2d += np.random.randn(23, 2) * 3
    return points_2d


def run_mmpose_inference(img, config_file, checkpoint_file):
    """Runs MMPose inference to get 2D keypoints."""
    try:
        from mmpose.apis import inference_topdown, init_model
        from mmpose.structures import merge_data_samples

        print(f"[Info] Loading MMPose model: {os.path.basename(checkpoint_file)}...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = init_model(config_file, checkpoint_file, device=device)

        result = inference_topdown(model, img)
        pred_instances = merge_data_samples(result).pred_instances

        if hasattr(pred_instances, 'keypoints') and len(pred_instances.keypoints) > 0:
            return pred_instances.keypoints[0], pred_instances.keypoint_scores[0]
        else:
            print("[Warning] No keypoints detected by MMPose.")
            return None, None

    except ImportError:
        print("[Error] MMPose is not installed. Please install it or run without config/checkpoint.")
        return None, None
    except Exception as e:
        print(f"[Error] MMPose inference failed: {e}")
        return None, None


def visualize_results(img, points_2d, points_3d):
    """Visualizes 2D detection and 3D reconstruction side-by-side."""
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(14, 7))

    # --- 2D Plot ---
    ax1 = fig.add_subplot(121)
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.scatter(points_2d[:, 0], points_2d[:, 1], s=20, c='red', zorder=2)

    for u, v in OPTIMIZATION_SKELETON:
        if u < 23 and v < 23:
            ax1.plot([points_2d[u, 0], points_2d[v, 0]],
                     [points_2d[u, 1], points_2d[v, 1]], 'y-', linewidth=1.5, alpha=0.7)
    ax1.set_title("Input: 2D Keypoints")
    ax1.axis('off')

    # --- 3D Plot ---
    ax2 = fig.add_subplot(122, projection='3d')

    # Axis swapping for better visualization intuition
    plot_x = points_3d[:, 0]
    plot_y = points_3d[:, 2]  # Depth becomes Y
    plot_z = -points_3d[:, 1]  # Image Y flipped to become Z (Up)

    ax2.scatter(plot_x, plot_y, plot_z, c='blue', s=30)

    for u, v in OPTIMIZATION_SKELETON:
        if u < 23 and v < 23:
            ax2.plot([plot_x[u], plot_x[v]], [plot_y[u], plot_y[v]], [plot_z[u], plot_z[v]], 'r-', linewidth=2)

    # Annotations
    ax2.text(plot_x[0], plot_y[0], plot_z[0], "Head", color='black', fontweight='bold')
    ax2.text(plot_x[13], plot_y[13], plot_z[13], "Torso", color='black', fontweight='bold')

    # Set equal aspect ratio
    max_range = np.array(
        [plot_x.max() - plot_x.min(), plot_y.max() - plot_y.min(), plot_z.max() - plot_z.min()]).max() / 2.0
    mid_x, mid_y, mid_z = plot_x.mean(), plot_y.mean(), plot_z.mean()
    ax2.set_xlim(mid_x - max_range, mid_x + max_range)
    ax2.set_ylim(mid_y - max_range, mid_y + max_range)
    ax2.set_zlim(mid_z - max_range, mid_z + max_range)

    ax2.set_xlabel("X (Width)")
    ax2.set_ylabel("Z (Depth)")
    ax2.set_zlabel("Y (Height)")
    ax2.set_title("Output: 3D Reconstruction")
    ax2.view_init(elev=20, azim=-60)

    plt.tight_layout()
    plt.show()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="3D Bird Pose Lifter Demo")
    parser.add_argument('--image', type=str, required=True, help="Path to the input image file.")
    parser.add_argument('--config', type=str, default=None, help="Path to MMPose config file (optional).")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to MMPose checkpoint file (optional).")

    args = parser.parse_args()

    # 1. Load Image
    img = load_image(args.image)
    h, w = img.shape[:2]

    # 2. Get 2D Keypoints
    points_2d = None
    if args.config and args.checkpoint:
        points_2d, scores = run_mmpose_inference(img, args.config, args.checkpoint)

    if points_2d is None:
        print("[Info] Using dummy 2D data.")
        points_2d = generate_dummy_2d(w, h)

    # 3. Run 3D Optimization
    print("[Info] Running 3D optimization...")
    lifter = BirdPoseLifter()
    points_3d = lifter.optimize_3d(points_2d, w, h)

    # 4. Visualize
    visualize_results(img, points_2d, points_3d)


if __name__ == "__main__":
    main()