import numpy as np
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PFLDLoss(nn.Module):
    def __init__(self, mode= 'wing'):
        super(PFLDLoss, self).__init__()
        self.mode = mode
    def forward(self, attribute_gt, landmark_gt, euler_angle_gt, angle, landmarks, train_batchsize, euler_angle_weight):  # mode: loss类型，l2,wing两种
        # euler_weight = 10
        weight_angle = euler_angle_weight * torch.sum(1 - torch.cos(angle - (euler_angle_gt * np.pi / 180.0)) + 0.000000001, axis=1)
        # weight_angle = euler_angle_weight * torch.sum(torch.abs(angle - euler_angle_gt), axis=1)


        # print(*weight_angle)
        # print('=========================================>')


        # print('euler angle loss: ', torch.mean(weight_angle))
        attributes_w_n = attribute_gt[:, 1:6].float()
        mat_ratio = torch.mean(attributes_w_n, axis=0)
        mat_ratio = torch.Tensor([1.0 / (x) if x > 0 else train_batchsize for x in mat_ratio]).to(device)
        weight_attribute = torch.sum(attributes_w_n.mul(mat_ratio), axis=1)
        # print('attribute weight:', torch.mean(weight_attribute))
        if self.mode == 'l2':
            loss = torch.sum((landmark_gt - landmarks) ** 2, axis=1)   # (landmark_gt - landmarks)
        else:
            loss = wing_loss(landmark_gt, landmarks, N_LANDMARK = 17)

        # indices = torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 32, 33]).to(device)
        # landmark_eye_gt = torch.index_select(landmark_gt, 1, indices)
        # landmarks_eye = torch.index_select(landmarks, 1, indices)

        # l2_distant_aux = torch.sum((landmark_eye_gt - landmarks_eye) ** 2, axis=1)

        # print('landmark loss: ', torch.mean(l2_distant))
        # return torch.mean(weight_angle * weight_attribute * l2_distant), torch.mean(l2_distant)
        # return (torch.mean(weight_angle * weight_attribute) + torch.mean(l2_distant * weight_attribute)), torch.mean(l2_distant)
        return torch.mean(loss * weight_attribute), torch.mean(loss)


def smoothL1(y_true, y_pred, beta = 1/9):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    mae = torch.abs(y_true - y_pred)
    loss = torch.sum(torch.where(mae>beta, mae-0.5*beta , 0.5*mae**2/beta), axis=-1)
    return torch.mean(loss)

def wing_loss(y_true, y_pred, w=10.0, epsilon=2.0, N_LANDMARK = 98):
    y_pred = y_pred.reshape(-1, N_LANDMARK, 2)
    y_true = y_true.reshape(-1, N_LANDMARK, 2) 
    
    x = y_true - y_pred
    c = w * (1.0 - np.math.log(1.0 + w / epsilon))
    absolute_x = torch.abs(x)
    losses = torch.where(w > absolute_x, w * torch.log(1.0 + absolute_x/epsilon), absolute_x - c)
    loss = torch.mean(torch.sum(losses, axis=[1, 2]), axis=0)
    return loss