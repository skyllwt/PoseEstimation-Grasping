from typing import Tuple, Dict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ..config import Config
from ..vis import Vis



class EstCoordNet(nn.Module):

    config: Config

    def __init__(self, config: Config):
        """
        Estimate the coordinates in the object frame for each object point.
        """
        super().__init__()
        self.config = config

        # Initial shared MLPs, grouped in Sequential
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )


        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )


        self.segmentation_head = nn.Sequential(
            nn.Conv1d(64 + 1024, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 3, 1), 
        )

    def forward(
        self, pc: torch.Tensor, coord: torch.Tensor, **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """
        Forward of EstCoordNet

        Parameters
        ----------
        pc: torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)
        coord: torch.Tensor
            Ground truth coordinates in the object frame, shape \(B, N, 3\)

        Returns
        -------
        float
            The loss value according to ground truth coordinates
        Dict[str, float]
            A dictionary containing additional metrics you want to log
        """
        B, N, _ = pc.shape
        pc = pc.transpose(1, 2)
        point_feature = self.mlp1(pc) 
        point_global_feature = self.mlp2(point_feature) 
        global_feature, _ = torch.max(point_global_feature, dim=2) 
        global_feature_expanded = global_feature.unsqueeze(-1).repeat(1, 1, N) 
        concat_features = torch.cat([point_feature, global_feature_expanded], dim=1)
        est_coord = self.segmentation_head(concat_features) 
        est_coord = est_coord.transpose(1, 2)
        loss = F.mse_loss(est_coord, coord)
        metric = dict(
            loss=loss,        
        )
        return loss, metric

    def est(self, pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate translation and rotation in the camera frame

        Parameters
        ----------
        pc : torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)

        Returns
        -------
        trans: torch.Tensor
            Estimated translation vector in camera frame, shape \(B, 3\)
        rot: torch.Tensor
            Estimated rotation matrix in camera frame, shape \(B, 3, 3\)

        Note
        ----
        The rotation matrix should satisfy the requirement of orthogonality and determinant 1.

        We don't have a strict limit on the running time, so you can use for loops and numpy instead of batch processing and torch.

        The only requirement is that the input and output should be torch tensors on the same device and with the same dtype.
        """
        device = pc.device
        dtype = pc.dtype
        B, N, _ = pc.shape

        # 1. Neural Network Inference to get estimated object coordinates
        # This part is essential and cannot be changed without modifying the network itself
        original_camera_points = pc.clone() # Rename for clarity
        pc_transposed = pc.transpose(1, 2)
        with torch.no_grad(): # Disable gradient calculation
            self.eval() # Set model to evaluation mode
            point_features = self.mlp1(pc_transposed)
            point_global_features = self.mlp2(point_features)
            global_features, _ = torch.max(point_global_features, dim=2)
            global_features_expanded = global_features.unsqueeze(-1).repeat(1, 1, N)
            combined_features = torch.cat([point_features, global_features_expanded], dim=1)
            estimated_object_coords = self.segmentation_head(combined_features).transpose(1, 2) # (B, N, 3)
            # self.train() # Optional: set model back to training mode if needed later

        # 2. Robust Pose Estimation using MSAC (variant of RANSAC)

        num_ransac_iterations = 1500 # Increased iterations for potentially better robustness
        distance_threshold = 0.03 # Threshold for considering a point an 'inlier'
        sample_set_size = 3 # Minimum number of points needed to estimate a pose

        # MSAC parameters
        msac_threshold_sq_tensor = torch.tensor(distance_threshold ** 2, device=device, dtype=dtype) # Squared threshold as tensor
        # The MSAC score is the sum of min(error_sq, threshold_sq)

        best_msac_scores = torch.full((B,), float('inf'), device=device, dtype=dtype)
        best_rotation_matrix = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
        best_translation_vector = torch.zeros(B, 3, device=device, dtype=dtype)
        best_inlier_masks = torch.zeros(B, N, dtype=torch.bool, device=device)


        for iter_idx in range(num_ransac_iterations):
            # Select a random subset of points for each batch item
            # Use torch.randint allowing replacement, common in RANSAC variants
            try:
                 # Ensure distinct indices if possible, though with replacement is simpler and standard
                 # For distinct indices without batching: torch.randperm(N)[:sample_set_size]
                 # Batched distinct indices is more complex, sticking to simple sampling for clarity/speed
                 # Let's try sampling without replacement using torch.randperm per batch item
                 sampled_indices = torch.stack([torch.randperm(N, device=device)[:sample_set_size] for _ in range(B)]) # (B, sample_set_size)
            except Exception as e:
                 # Handle case where N < sample_set_size if necessary, though unlikely with point clouds
                 print(f"Error sampling points: {e}")
                 if N < sample_set_size:
                     print(f"Warning: Number of points ({N}) is less than sample size ({sample_set_size}). Skipping iteration.")
                     continue # Skip iteration if not enough points


            # Extract corresponding points from the subset
            cam_points_subset = torch.gather(original_camera_points, 1, sampled_indices.unsqueeze(-1).expand(-1, -1, 3)) # (B, k, 3)
            obj_coords_subset = torch.gather(estimated_object_coords, 1, sampled_indices.unsqueeze(-1).expand(-1, -1, 3)) # (B, k, 3)

            # Calculate centroids for the subset
            centroid_cam = cam_points_subset.mean(dim=1, keepdim=True) # (B, 1, 3)
            centroid_obj = obj_coords_subset.mean(dim=1, keepdim=True) # (B, 1, 3)

            # Center the point subsets
            centered_cam_subset = cam_points_subset - centroid_cam # (B, k, 3)
            centered_obj_subset = obj_coords_subset - centroid_obj # (B, k, 3)

            # Compute the cross-covariance matrix H
            # H = sum(p'_i * q'_i^T) for p' = R q' (aligning object to camera frame)
            # Dimensions: (B, 3, k) @ (B, k, 3) -> (B, 3, 3)
            cross_covariance_H = torch.bmm(centered_cam_subset.transpose(1, 2), centered_obj_subset)

            # Perform SVD on H
            try:
                 # Use linalg.svd for batched SVD
                 U_svd, S_svd, Vh_svd = torch.linalg.svd(cross_covariance_H) # Vh_svd is V^T

                 # Calculate the rotation matrix candidate R
                 # R = U V^T from H = U S V^T where H = sum(p'q'^T)
                 rotation_candidate = torch.bmm(U_svd, Vh_svd) # (B, 3, 3)

                 # Ensure rotation matrix is proper (determinant +1)
                 # Check determinant
                 det_candidate = torch.det(rotation_candidate) # (B,)

                 # Create a diagonal matrix to fix reflection
                 # If det is -1, flip the sign of the last column of V before multiplying
                 # V = Vh_svd.transpose(1, 2)
                 # V_fixed = V.clone()
                 # V_fixed[:, :, 2] *= det_candidate.sign().unsqueeze(-1)
                 # rotation_candidate_fixed = torch.bmm(U_svd, V_fixed.transpose(1, 2))

                 # A more direct way using the diagonal matrix in SVD reconstruction
                 # R = U @ diag([1, 1, det(UV^T)]) @ V^T
                 reflection_corrector = torch.diag(torch.tensor([1., 1., 1.], device=device, dtype=dtype)).unsqueeze(0).repeat(B, 1, 1)
                 reflection_corrector[:, 2, 2] = det_candidate.sign()
                 rotation_candidate_fixed = torch.bmm(U_svd, torch.bmm(reflection_corrector, Vh_svd))


                 # Calculate the translation vector candidate T
                 # T = p_bar - R @ q_bar
                 translation_candidate = centroid_cam.squeeze(1) - torch.bmm(rotation_candidate_fixed, centroid_obj.transpose(1, 2)).squeeze(-1) # (B, 3)

            except RuntimeError as e:
                # SVD might fail for degenerate point sets (e.g., collinear)
                # print(f"Warning: SVD failed during RANSAC iteration {iter_idx}. Skipping hypothesis. Error: {e}")
                continue # Skip this iteration if SVD fails


            # Evaluate hypothesis: Transform ALL estimated object points back to camera frame
            # pc_pred = R @ est_coord + T
            transformed_obj_coords = torch.bmm(rotation_candidate_fixed, estimated_object_coords.transpose(1, 2)).transpose(1, 2) + translation_candidate.unsqueeze(1) # (B, N, 3)

            # Calculate squared errors for all points
            all_point_errors_sq = torch.sum((original_camera_points - transformed_obj_coords) ** 2, dim=2) # (B, N)

            # Calculate MSAC score for the current hypothesis
            # Score = sum(min(error_sq, threshold_sq))
            current_msac_score = torch.sum(torch.minimum(all_point_errors_sq, msac_threshold_sq_tensor), dim=1) # (B,)

            # Identify potential inliers for this hypothesis
            current_inlier_mask = all_point_errors_sq < msac_threshold_sq_tensor # (B, N)

            # Update the best hypothesis for batches where the current score is better
            update_mask = current_msac_score < best_msac_scores # (B,)

            best_msac_scores[update_mask] = current_msac_score[update_mask]
            best_rotation_matrix[update_mask] = rotation_candidate_fixed[update_mask]
            best_translation_vector[update_mask] = translation_candidate[update_mask]
            best_inlier_masks[update_mask] = current_inlier_mask[update_mask]


        # 3. Refine the pose using ALL points identified as inliers by the best hypothesis

        final_rotation = best_rotation_matrix.clone()
        final_translation = best_translation_vector.clone()

        # Refine pose for each batch item
        for i in range(B):
            inlier_indices = torch.where(best_inlier_masks[i])[0]

            # Only refine if we have enough inliers
            if len(inlier_indices) >= sample_set_size:
                cam_inliers = original_camera_points[i, inlier_indices, :] # (num_inliers, 3)
                obj_inlier_coords = estimated_object_coords[i, inlier_indices, :] # (num_inliers, 3)

                # Recalculate centroids and center points for inliers
                inlier_centroid_cam = cam_inliers.mean(dim=0, keepdim=True) # (1, 3)
                inlier_centroid_obj = obj_inlier_coords.mean(dim=0, keepdim=True) # (1, 3)

                centered_cam_inliers = cam_inliers - inlier_centroid_cam # (num_inliers, 3)
                centered_obj_inlier_coords = obj_inlier_coords - inlier_centroid_obj # (num_inliers, 3)

                # Compute cross-covariance matrix for inliers
                # (3, num_inliers) @ (num_inliers, 3) -> (3, 3)
                inlier_cross_covariance_H = centered_cam_inliers.transpose(0, 1) @ centered_obj_inlier_coords

                # Perform SVD for refinement
                try:
                    U_refine, S_refine, Vh_refine = torch.linalg.svd(inlier_cross_covariance_H)

                    # Calculate refined rotation R = U V^T
                    refined_R = U_refine @ Vh_refine

                    # Ensure proper rotation (determinant +1)
                    det_refined_R = torch.det(refined_R)
                    # V_refine = Vh_refine.transpose(0, 1)
                    # V_refine_fixed = V_refine.clone()
                    # V_refine_fixed[:, 2] *= det_refined_R.sign()
                    # refined_R_fixed = U_refine @ V_refine_fixed.transpose(0, 1)
                    reflection_corrector_refine = torch.diag(torch.tensor([1., 1., det_refined_R.sign()], device=device, dtype=dtype))
                    refined_R_fixed = U_refine @ reflection_corrector_refine @ Vh_refine


                    # Calculate refined translation T = p_bar - R @ q_bar
                    refined_T = inlier_centroid_cam.squeeze(0) - refined_R_fixed @ inlier_centroid_obj.squeeze(0)

                    # Update final pose for this batch item
                    final_rotation[i] = refined_R_fixed
                    final_translation[i] = refined_T

                except RuntimeError as e:
                    # SVD might fail even during refinement (e.g., inliers are collinear)
                    print(f"Warning: SVD failed during refinement for batch item {i}. Keeping RANSAC result. Error: {e}")
                    # The best RANSAC result is already in final_rotation/translation for this item


        # Return the estimated translation and rotation
        return final_translation, final_rotation