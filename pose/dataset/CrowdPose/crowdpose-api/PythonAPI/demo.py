from crowdposetools.coco import COCO
from crowdposetools.cocoeval import COCOeval

gt_file = '../annotations/crowdpose_val.json'
preds = '/home/hrnet_br/1014_UDP_1102/output/cocoac_c/pose_hrnet/local/results/keypoints_val_results_0.json'

cocoGt = COCO(gt_file)
cocoDt = cocoGt.loadRes(preds)
cocoEval = COCOeval(cocoGt, cocoDt, 'keypoints')
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
