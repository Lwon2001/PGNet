import torch
import torch.nn as nn
import chamfer

class F1Score(nn.Module):
    "ref: https://github.com/diegovalsesia/XMFnet/blob/main/eval.py"
    def __init__(self):
        super(F1Score, self).__init__()
    
    def forward(self, array1, array2, threshold=0.001):
        """
        Calculates the F1-Score, Precision, and Recall between two point clouds.
        
        Args:
            array1 (torch.Tensor): Predicted point clouds (B, N, 3)
            array2 (torch.Tensor): Ground truth point clouds (B, M, 3)
            threshold (float): Distance threshold for a point to be considered correct.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing
                - fscore (B,): The F1 score for each element in the batch.
                - precision (B,): The precision for each element in the batch.
                - recall (B,): The recall for each element in the batch.
        """
        dist1, dist2, _, _ = chamfer.forward(array1, array2)
        precision = torch.mean((dist1 < threshold).float(), dim=1)
        recall = torch.mean((dist2 < threshold).float(), dim=1)
        fscore = 2 * precision * recall / (precision + recall)
        fscore[torch.isnan(fscore)] = 0
        return fscore, precision, recall


