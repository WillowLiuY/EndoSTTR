import numpy as np
import cv2
from .pfm_Q import load_pfm

def create_point_cloud(disparity_map, left_image, Q, reliability_mask):
    if disparity_map.dtype != np.float32:
        disparity_map = disparity_map.astype(np.float32)

    Q = np.array(Q, dtype=np.float32)
    if Q.shape != (4, 4):
        raise ValueError("Q matrix must be a 4x4 matrix.")

    filtered_disparity = np.zeros_like(disparity_map)
    filtered_disparity[reliability_mask] = disparity_map[reliability_mask]

    points_3D = cv2.reprojectImageTo3D(filtered_disparity, Q)

    mask = filtered_disparity > 0
    output_points = points_3D[mask]
    colors = left_image[mask]
    return output_points, colors

def compute_depth_error(disp_estimated, gt_disp_path, Q_matrix, valid_mask, occ_mask):
    gt_disp, _ = load_pfm(gt_disp_path)

    focal_length = Q_matrix[2][3]
    baseline = 1 / Q_matrix[3][2]

    def safe_divide(a, b):
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.true_divide(a, b)
            c[~np.isfinite(c)] = 0
        return c

    depth_ground_truth = safe_divide(focal_length * baseline, gt_disp)

    occ_mask_resized = cv2.resize(occ_mask.astype(np.float32), (disp_estimated.shape[1], disp_estimated.shape[0])) > 0.5
    combined_mask = np.logical_and(valid_mask, ~occ_mask_resized)

    valid_disp_estimated = disp_estimated[combined_mask]
    valid_depth_ground_truth = depth_ground_truth[combined_mask]

    valid_depth_estimated = safe_divide(focal_length * baseline, valid_disp_estimated)

    depth_error = np.abs(valid_depth_estimated - valid_depth_ground_truth)

    return depth_error