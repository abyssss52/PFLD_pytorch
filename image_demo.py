import argparse
import time
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simps

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from dataset.datasets import WLFWDatasets
from dataset.datasets import WLFWDatasets_image_demo

from models.pfld import PFLDInference, AuxiliaryNet

cudnn.benchmark = True
cudnn.determinstic = True
cudnn.enabled = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_inference(loader, plfd_backbone, auxiliary_backbone):
    plfd_backbone.eval()

    # nme_list = []
    cost_time = 0
    with torch.no_grad():
        start_time = time.time()
        for img in loader:
            img = img.to(device)
            print(img.size())
            plfd_backbone = plfd_backbone.to(device)

            feature, landmarks = plfd_backbone(img)
            angle = auxiliary_backbone(feature)
            cost_time = time.time() - start_time
            print('耗时：', cost_time)
            print('euler angle is ', ((angle * 180) / np.pi))

            landmarks = landmarks.cpu().numpy()
            landmarks = landmarks.reshape(landmarks.shape[0], -1, 2) # landmark

            if args.show_image:
                print(img.cpu().numpy().shape)
                show_img = np.array(np.transpose(img[0].cpu().numpy(), (1, 2, 0)))
                show_img = (show_img * 255).astype(np.uint8)
                np.clip(show_img, 0, 255)

                pre_landmark = landmarks[0] * [112, 112]
                # print(pre_landmark)
                cv2.imwrite("xxx.jpg", show_img)
                img_clone = cv2.imread("xxx.jpg")
                idx = 0
                for (x, y) in pre_landmark.astype(np.int32):
                    cv2.circle(img_clone, (x, y), 1, (255,0,0),-1)
                    # cv2.putText(img_clone, str(idx), (x+1,y+1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.3, color=(0,0,0), thickness=1)
                    idx += 1
                cv2.imshow("result", img_clone)
                if cv2.waitKey(0) == ord('n'):
                    break

        #     nme_temp = compute_nme(landmarks, landmark_gt)
        #     for item in nme_temp:
        #         nme_list.append(item)
        #
        # # nme
        # print('nme: {:.4f}'.format(np.mean(nme_list)))
        # # auc and failure rate
        # failureThreshold = 0.1
        # auc, failure_rate = compute_auc(nme_list, failureThreshold)
        # print('auc @ {:.1f} failureThreshold: {:.4f}'.format(failureThreshold, auc))
        # print('failure_rate: {:}'.format(failure_rate))
        # # inference time
        # print("inference_cost_time: {0:4f}".format(np.mean(cost_time)))

def main(args):
    checkpoint = torch.load(args.model_path, map_location=device)
    plfd_backbone = PFLDInference().to(device)
    auxiliary_backbone = AuxiliaryNet().to(device)
    plfd_backbone.load_state_dict(checkpoint['plfd_backbone'])
    auxiliary_backbone.load_state_dict(checkpoint['auxiliarynet'])

    transform = transforms.Compose([transforms.ToTensor()])
    image_path = '/home/night/图片/temp/face_mask/'#'/home/night/PycharmProjects/face/head_pose_estimation/PFLD-pytorch/data/WFLW/WFLW_images/40--Gymnastics'#
    for img in os.listdir(image_path):
        image_test = WLFWDatasets_image_demo(os.path.join(image_path,img), transform)   # args.test_image, transform
        # wlfw_val_dataset = WLFWDatasets(args.test_dataset, transform)
        image_test_dataloader = DataLoader(image_test, batch_size=1, shuffle=False, num_workers=0)
        image_inference(image_test_dataloader, plfd_backbone, auxiliary_backbone)

def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_path', default="./checkpoint/snapshot/epoch_70.pth.tar", type=str)   # "./checkpoint/snapshot/checkpoint.pth.tar"    epoch_404.pth.tar
    parser.add_argument('--test_image', default='/home/night/图片/temp/face_mask/IMG_20200331_103908.jpg',type=str)   # '/home/night/PycharmProjects/face/head_pose_estimation/PFLD-pytorch/data/test_data/imgs/8_9_Press_Conference_Press_Conference_9_477_0.png'
    # parser.add_argument('--test_dataset', default='./data/test_data/list.txt', type=str)
    parser.add_argument('--show_image', default=True, type=bool)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)