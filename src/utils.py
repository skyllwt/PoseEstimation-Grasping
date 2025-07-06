import os
import random
from typing import Tuple, Optional
import numpy as np
import torch
from transforms3d.quaternions import quat2mat

from .type import Grasp
from .constants import PC_MAX, PC_MIN


def to_pose(
    trans: Optional[np.ndarray] = None, rot: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert translation and rotation into a 4x4 pose matrix.

    Parameters
    ----------
    trans: Optional[np.ndarray]
        Translation vector, shape (3,).
    rot: Optional[np.ndarray]
        Rotation matrix, shape (3, 3).

    Returns
    -------
    np.ndarray
        4x4 pose matrix.
    """
    ret = np.eye(4)
    if trans is not None:
        ret[:3, 3] = trans
    if rot is not None:
        ret[:3, :3] = rot
    return ret


def set_seed(seed: int):
    """
    Set random seed for reproducibility.

    Parameters
    ----------
    seed: int
        Random seed between 0 and 2**32 - 1
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)



def transform_grasp_pose(
    grasp: Grasp,
    est_trans: np.ndarray,
    est_rot: np.ndarray,
    cam_trans: np.ndarray,
    cam_rot: np.ndarray,
) -> Grasp:
    """
    Transform grasp from the object frame into the robot frame

    Parameters
    ----------
    grasp: Grasp
        The grasp to be transformed.
    est_trans: np.ndarray
        Estimated translation vector in the camera frame.
    est_rot: np.ndarray
        Estimated rotation matrix in the camera frame.
    cam_trans: np.ndarray
        Camera's translation vector in the robot frame.
    cam_rot: np.ndarray
        Camera's rotation matrix in the robot frame.

    Returns
    -------
    Grasp
        The transformed grasp in the robot frame.
    """
    # 1. Grasp pose in object frame (T_grasp_to_object)
    # This is the transformation from the grasp frame to the object frame
    T_grasp_to_object = to_pose(trans=grasp.trans, rot=grasp.rot)

    # 2. Estimated object pose in camera frame (T_object_to_camera)
    # This is the transformation from the object frame to the camera frame
    T_object_to_camera = to_pose(trans=est_trans, rot=est_rot)

    # 3. Camera pose in robot frame (T_camera_to_robot)
    # This is the transformation from the camera frame to the robot frame
    T_camera_to_robot = to_pose(trans=cam_trans, rot=cam_rot)

    # 4. Compute the combined transformation matrix from grasp frame to robot frame
    # T_grasp_to_robot = T_camera_to_robot @ T_object_to_camera @ T_grasp_to_object
    # Using np.dot for clarity and compatibility
    T_grasp_to_robot = np.dot(T_camera_to_robot, np.dot(T_object_to_camera, T_grasp_to_object))

    # 5. Extract the transformed translation and rotation
    transformed_trans = T_grasp_to_robot[:3, 3] # The last column of the 3x4 part
    transformed_rot = T_grasp_to_robot[:3, :3] # The top-left 3x3 part

    # 6. Update the input grasp object's pose attributes with the transformed pose
    # We assume the Grasp object is mutable and has 'trans' and 'rot' attributes that are np.ndarray
    grasp.trans = transformed_trans
    grasp.rot = transformed_rot

    # 7. Return the modified grasp object
    return grasp

def get_pc(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """
    Convert depth image into point cloud using intrinsics

    All points with depth=0 are filtered out

    Parameters
    ----------
    depth: np.ndarray
        Depth image, shape (H, W)
    intrinsics: np.ndarray
        Intrinsics matrix with shape (3, 3)

    Returns
    -------
    np.ndarray
        Point cloud with shape (N, 3)
    """
    # Get image dimensions
    height, width = depth.shape
    # Create meshgrid for pixel coordinates
    v, u = np.meshgrid(range(height), range(width), indexing="ij")
    # Flatten the arrays
    u = u.flatten()
    v = v.flatten()
    depth_flat = depth.flatten()
    # Filter out invalid depth values
    valid = depth_flat > 0
    u = u[valid]
    v = v[valid]
    depth_flat = depth_flat[valid]
    # Create homogeneous pixel coordinates
    pixels = np.stack([u, v, np.ones_like(u)], axis=0)
    # Convert pixel coordinates to camera coordinates
    rays = np.linalg.inv(intrinsics) @ pixels
    # Scale rays by depth
    points = rays * depth_flat
    return points.T


def get_workspace_mask(pc: np.ndarray) -> np.ndarray:
    """Get the mask of the point cloud in the workspace."""
    pc_mask = (
        (pc[:, 0] > PC_MIN[0])
        & (pc[:, 0] < PC_MAX[0])
        & (pc[:, 1] > PC_MIN[1])
        & (pc[:, 1] < PC_MAX[1])
        & (pc[:, 2] > PC_MIN[2])
        & (pc[:, 2] < PC_MAX[2])
    )
    return pc_mask


def rand_rot_mat() -> np.ndarray:
    """
    Generate a random rotation matrix with shape (3, 3) uniformly.
    """
    while True:
        quat = np.random.randn(4)
        if np.linalg.norm(quat) > 1e-6:
            break
    quat /= np.linalg.norm(quat)
    return quat2mat(quat)


def theta_to_2d_rot(theta: float) -> np.ndarray:
    """
    Convert a 2D rotation angle into a rotation matrix.

    Parameters
    ----------
    theta : float
        The rotation angle in radians.

    Returns
    -------
    np.ndarray
        The resulting 2D rotation matrix (2, 2).
    """
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def rot_dist(r1: np.ndarray, r2: np.ndarray) -> float:
    """
    The relative rotation angle between two rotation matrices.

    Parameters
    ----------
    r1 : np.ndarray
        The first rotation matrix (3, 3).
    r2 : np.ndarray
        The second rotation matrix (3, 3).

    Returns
    -------
    float
        The relative rotation angle in radians.
    """
    return np.arccos(np.clip((np.trace(r1 @ r2.T) - 1) / 2, -1, 1))


# You can add additional functions here
