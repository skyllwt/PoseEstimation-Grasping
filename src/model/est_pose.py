from typing import Tuple, Dict
import torch
from torch import nn

from ..config import Config


class EstPoseNet(nn.Module):

    config: Config

    def __init__(self, config: Config):
        """
        Directly estimate the translation vector and rotation matrix.
        """
        super().__init__()
        self.config = config

        ###

        self.conv1=nn.Conv1d(3,64,1)
        self.conv2=nn.Conv1d(64,128,1)
        self.conv3=nn.Conv1d(128,1024,1)
        self.bn1=nn.BatchNorm1d(64)
        self.bn2=nn.BatchNorm1d(128)
        self.bn3=nn.BatchNorm1d(1024)
        self.relu=nn.ReLU()

        # Separate fully connected layers for translation and rotation
        self.fc1_trans = nn.Linear(1024, 256)  # For translation
        self.fc2_trans = nn.Linear(256, 128)  # For translation
        self.bn_fc1_trans = nn.BatchNorm1d(256)
        self.bn_fc2_trans = nn.BatchNorm1d(128)

        self.fc1_rot = nn.Linear(1024, 256)  # For rotation
        self.fc2_rot = nn.Linear(256, 128)  # For rotation
        self.bn_fc1_rot = nn.BatchNorm1d(256)
        self.bn_fc2_rot = nn.BatchNorm1d(128)

        # Separate dropout layers for translation and rotation
        self.dropout_trans = nn.Dropout(0.3)  # For translation
        self.dropout_rot = nn.Dropout(0.3)  # For rotation

        # Separate heads for translation and rotation
        self.trans_head = nn.Linear(128, 3)  # For predicting translation vector
        self.rot_head = nn.Linear(128, 9)  # For predicting 9D rotation representation



    def forward(
        self, pc: torch.Tensor, trans: torch.Tensor, rot: torch.Tensor, **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """
        Forward of EstPoseNet

        Parameters
        ----------
        pc : torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)
        trans : torch.Tensor
            Ground truth translation vector in camera frame, shape \(B, 3\)
        rot : torch.Tensor
            Ground truth rotation matrix in camera frame, shape \(B, 3, 3\)

        Returns
        -------
        float
            The loss value according to ground truth translation and rotation
        Dict[str, float]
            A dictionary containing additional metrics you want to log
        """


        # PointNet backbone
        x = pc.transpose(1, 2)  # (B, N, 3) -> (B, 3, N)
        x = self.relu(self.bn1(self.conv1(x)))  # (B, 64, N)
        x = self.relu(self.bn2(self.conv2(x)))  # (B, 128, N)
        x = self.relu(self.bn3(self.conv3(x)))  # (B, 1024, N)

        # Global max pooling
        x = torch.max(x, dim=2)[0]  # (B, 1024)

        # Separate fully connected layers for translation and rotation
        # Translation network path
        x_trans = self.relu(self.bn_fc1_trans(self.fc1_trans(x)))  # (B, 256)
        x_trans = self.dropout_trans(x_trans)  # Apply dropout for translation
        x_trans = self.relu(self.bn_fc2_trans(self.fc2_trans(x_trans)))  # (B, 128)
        x_trans = self.dropout_trans(x_trans)  # Apply dropout for translation

        # Rotation network path
        x_rot = self.relu(self.bn_fc1_rot(self.fc1_rot(x)))  # (B, 256)
        x_rot = self.dropout_rot(x_rot)  # Apply dropout for rotation
        x_rot = self.relu(self.bn_fc2_rot(self.fc2_rot(x_rot)))  # (B, 128)
        x_rot = self.dropout_rot(x_rot)  # Apply dropout for rotation


        # Separate heads for translation and rotation
        pred_trans = self.trans_head(x_trans)  # (B, 3)
        pred_rot_9d = self.rot_head(x_rot)  # (B, 9)

        # Convert 9D representation to rotation matrix using SVD
        pred_rot = self._orthogonalize_9d_to_matrix(pred_rot_9d)  # (B, 3, 3)

        # Compute losses
        # Translation loss: MSE
        trans_loss = torch.mean((pred_trans - trans) ** 2)

        rot_loss = torch.mean((pred_rot_9d - rot.view(rot.size(0), -1)) ** 2)

        # Total loss with weighting
        loss =0.2*trans_loss + 0.8*rot_loss

        # Metrics
        metric = {
            'loss': loss,
            'trans_loss': trans_loss,
            'rot_loss': rot_loss,
            'mean_trans_error': torch.mean(torch.norm(pred_trans - trans, dim=1)),
            'mean_rot_error': rot_loss,
        }

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
        """
        # PointNet backbone
        x = pc.transpose(1, 2)  # (B, N, 3) -> (B, 3, N)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        # Global max pooling
        x = torch.max(x, dim=2)[0]  # (B, 1024)

        # Separate fully connected layers for translation and rotation
        # Translation network path
        x_trans = self.relu(self.bn_fc1_trans(self.fc1_trans(x)))  # (B, 256)
        x_trans = self.dropout_trans(x_trans)  # Apply dropout for translation
        x_trans = self.relu(self.bn_fc2_trans(self.fc2_trans(x_trans)))  # (B, 128)
        x_trans = self.dropout_trans(x_trans)  # Apply dropout for translation

        # Rotation network path
        x_rot = self.relu(self.bn_fc1_rot(self.fc1_rot(x)))  # (B, 256)
        x_rot = self.dropout_rot(x_rot)  # Apply dropout for rotation
        x_rot = self.relu(self.bn_fc2_rot(self.fc2_rot(x_rot)))  # (B, 128)
        x_rot = self.dropout_rot(x_rot)  # Apply dropout for rotation

        # Output heads
        trans = self.trans_head(x_trans)  # (B, 3)
        rot_9d = self.rot_head(x_rot)  # (B, 9)

        rot=self._orthogonalize_9d_to_matrix(rot_9d)

        return trans,rot

    def _quaternion_to_matrix(self, quat: torch.Tensor) -> torch.Tensor:
        """
        Convert quaternion to rotation matrix.

        Parameters
        ----------
        quat : torch.Tensor
            Quaternion, shape (B, 4), [w, x, y, z]

        Returns
        -------
        rot : torch.Tensor
            Rotation matrix, shape (B, 3, 3)
        """
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        B = quat.size(0)

        xx, yy, zz = x * x, y * y, z * z
        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z

        rot = torch.zeros(B, 3, 3, device=quat.device)
        rot[:, 0, 0] = 1 - 2 * (yy + zz)
        rot[:, 0, 1] = 2 * (xy - wz)
        rot[:, 0, 2] = 2 * (xz + wy)
        rot[:, 1, 0] = 2 * (xy + wz)
        rot[:, 1, 1] = 1 - 2 * (xx + zz)
        rot[:, 1, 2] = 2 * (yz - wx)
        rot[:, 2, 0] = 2 * (xz - wy)
        rot[:, 2, 1] = 2 * (yz + wx)
        rot[:, 2, 2] = 1 - 2 * (xx + yy)

        return rot

    def _matrix_to_quaternion(self, rot: torch.Tensor) -> torch.Tensor:
        """
        Convert rotation matrix to quaternion.

        Parameters
        ----------
        rot : torch.Tensor
            Rotation matrix, shape (B, 3, 3)

        Returns
        -------
        quat : torch.Tensor
            Quaternion, shape (B, 4), [w, x, y, z]
        """
        B = rot.size(0)
        quat = torch.zeros(B, 4, device=rot.device)

        trace = rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2]
        mask = trace > 0
        s = torch.sqrt(trace[mask] + 1.0) * 2
        quat[mask, 0] = s / 4
        quat[mask, 1] = (rot[mask, 2, 1] - rot[mask, 1, 2]) / s
        quat[mask, 2] = (rot[mask, 0, 2] - rot[mask, 2, 0]) / s
        quat[mask, 3] = (rot[mask, 1, 0] - rot[mask, 0, 1]) / s

        # Handle cases where trace <= 0
        mask_neg = ~mask
        if mask_neg.any():
            neg_indices = torch.where(mask_neg)[0]  # Indices where trace <= 0
            rot_neg = rot[neg_indices]  # Subset of rotation matrices
            m = rot_neg.diagonal(dim1=1, dim2=2).argmax(dim=1)  # Max diagonal index
            for idx, i in enumerate(neg_indices):
                if m[idx] == 0:
                    s = torch.sqrt(1.0 + rot_neg[idx, 0, 0] - rot_neg[idx, 1, 1] - rot_neg[idx, 2, 2]) * 2
                    quat[i, 0] = (rot_neg[idx, 2, 1] - rot_neg[idx, 1, 2]) / s
                    quat[i, 1] = s / 4
                    quat[i, 2] = (rot_neg[idx, 0, 1] + rot_neg[idx, 1, 0]) / s
                    quat[i, 3] = (rot_neg[idx, 0, 2] + rot_neg[idx, 2, 0]) / s
                elif m[idx] == 1:
                    s = torch.sqrt(1.0 + rot_neg[idx, 1, 1] - rot_neg[idx, 0, 0] - rot_neg[idx, 2, 2]) * 2
                    quat[i, 0] = (rot_neg[idx, 0, 2] - rot_neg[idx, 2, 0]) / s
                    quat[i, 1] = (rot_neg[idx, 0, 1] + rot_neg[idx, 1, 0]) / s
                    quat[i, 2] = s / 4
                    quat[i, 3] = (rot_neg[idx, 1, 2] + rot_neg[idx, 2, 1]) / s
                else:
                    s = torch.sqrt(1.0 + rot_neg[idx, 2, 2] - rot_neg[idx, 0, 0] - rot_neg[idx, 1, 1]) * 2
                    quat[i, 0] = (rot_neg[idx, 1, 0] - rot_neg[idx, 0, 1]) / s
                    quat[i, 1] = (rot_neg[idx, 0, 2] + rot_neg[idx, 2, 0]) / s
                    quat[i, 2] = (rot_neg[idx, 1, 2] + rot_neg[idx, 2, 1]) / s
                    quat[i, 3] = s / 4

        return quat / torch.norm(quat, dim=1, keepdim=True)


    def _quaternion_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """
        Multiply two quaternions.

        Parameters
        ----------
        q1, q2 : torch.Tensor
            Quaternions, shape (B, 4), [w, x, y, z]

        Returns
        -------
        q : torch.Tensor
            Result quaternion, shape (B, 4)
        """
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        q = torch.zeros_like(q1)
        q[:, 0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        q[:, 1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        q[:, 2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        q[:, 3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return q

    def _quaternion_conjugate(self, q: torch.Tensor) -> torch.Tensor:
        """
        Compute the conjugate of a quaternion.

        Parameters
        ----------
        q : torch.Tensor
            Quaternion, shape (B, 4), [w, x, y, z]

        Returns
        -------
        q_conj : torch.Tensor
            Conjugate quaternion, shape (B, 4)
        """
        q_conj = q.clone()
        q_conj[:, 1:] = -q_conj[:, 1:]
        return q_conj

    def _orthogonalize_9d_to_matrix(self, rot_9d: torch.Tensor) -> torch.Tensor:
        """
        Convert 9D rotation representation to a rotation matrix using SVD.

        Parameters
        ----------
        rot_9d : torch.Tensor
            9D rotation representation, shape (B, 9)

        Returns
        -------
        rot : torch.Tensor
            Rotation matrix, shape (B, 3, 3)
        """
        B = rot_9d.size(0)
        # Normalize 9D vector to prevent extreme values
        rot_9d = rot_9d / (torch.norm(rot_9d, dim=1, keepdim=True) + 1e-8)

        # Reshape 9D vector to 3x3 matrix
        rot_mat = rot_9d.view(B, 3, 3)  # (B, 3, 3)

        # Perform SVD
        try:
            U, _, Vt = torch.svd(rot_mat)  # U: (B, 3, 3), Vt: (B, 3, 3)

            if torch.isnan(U).any() or torch.isnan(Vt).any():
                print(f"NaN detected in U or Vt after SVD: {U}, {Vt}")

        except RuntimeError:
            # Fallback to identity matrix if SVD fails
            print("Warning: SVD failed, returning identity matrices")
            return torch.eye(3, device=rot_9d.device).unsqueeze(0).repeat(B, 1, 1)


        # Compute rotation matrix: R = U * Vt
        rot = torch.matmul(U, Vt.transpose(1, 2))  # (B, 3, 3)

        # Check for NaN in rot
        if torch.isnan(rot).any():
            print("Warning: NaN detected in SVD output, returning identity matrices")
            return torch.eye(3, device=rot_9d.device).unsqueeze(0).repeat(B, 1, 1)

        # Ensure det(R) = 1
        det = torch.det(rot)
        mask = det < 0
        if mask.any():
            # Flip the third column of U for negative determinant cases
            U=U.clone()
            U[mask, :, 2] = -U[mask, :, 2]
            rot[mask] = torch.matmul(U[mask], Vt[mask].transpose(1, 2))

        return rot