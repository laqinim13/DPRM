import sys
import os
from pose import inference
from pose.smooth import pose_filter

class PoseEstimation(inference.PoseEstimation):
    def __init__(self,
                 config_file1,
                 model_file1=None,
                 target: str = "",
                 use_box=True,
                 threshold=0.5,
                 device="cuda:0"
                 ):
        super(PoseEstimation, self).__init__(config_file1, model_file1, target, threshold, device)
        self.threshold = threshold
        self.use_box = use_box

    def inference(self, bgr, box, threshold=0.1):
        """
        input_tensor = image_processing.image_normalization(image,
                                                             mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])
        input_tensor = input_tensor.transpose(2, 0, 1)  # [H0,W1,C2]-[C,H,W]
        input_tensor = torch.from_numpy(input_tensor)
        :param bgr:
        :param box:
        :param threshold:
        :return:
        """
        input_tensor, center, scale = self.pre_process(bgr, box)
        output = self.forward(input_tensor).cpu().numpy()
        kp_point, kp_score = self.post_process(output, center, scale, threshold)
        return kp_point, kp_score, self.skeleton