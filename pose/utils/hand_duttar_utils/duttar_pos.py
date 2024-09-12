import cv2
from pose.utils.hand_duttar_utils.duttar_demo import PoseEstimation

class Duttar_Pose():
    def __init__(self, device="cuda:0"):
        self.device = device
        # self.class_names = None
        # self.class_names = ["person"]
        # sys.path.insert(0, os.path.join(os.path.dirname(__file__), "yolov5"))
        model_file1 = "work_space/dutar/2_3/model/best_model_299_1.0000.pth"
        config_file1 = "work_space/dutar/2_3/dutar.yaml"
        target = "duttar"
        threshold = 0.2
        # weights = "H:/success_hand_projct/re_project_3_1/yolov5/runs/train/exp4/weights/best.pt"
        self.detector = PoseEstimation(model_file1=model_file1,  # model.pt path(s)
                                       config_file1=config_file1,
                                       target=target,
                                       threshold=threshold,
                                       device=device,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                                       )
    def detect(self, bgr, boxes, threshold=0.3):
        """
        :param bgr: bgr image
        :param boxes:  [xmin, ymin, xmax, ymax]
        """
        kp_points, kp_scores = [], []
        for box in boxes:
            points, scores, skeleton = self.detector.inference(bgr, box, threshold)
            kp_points.append(points)
            kp_scores.append(scores)
        return kp_points, kp_scores, skeleton




