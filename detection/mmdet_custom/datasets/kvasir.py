
import os.path as osp
import json
import numpy as np
import mmcv
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmdet.core import BitmapMasks

@DATASETS.register_module()
class KvasirDataset(CustomDataset):
    CLASSES = ('polyp',)

    def load_annotations(self, ann_file):
        data_infos = []
        with open(ann_file, 'r') as f:
            data = json.load(f)
            
        # Data format: "ID": {"height": H, "width": W, "bbox": [...]}
        for img_id, info in data.items():
            # Images in 'images/{id}.jpg' relative to data_root if data_root is Kvasir parent dir
            # But usually img_prefix in config will be 'data/Kvasir-SEG'
            # Images are in 'images/' subdir.
            
            # Check file extension
            # The IDs do not have extensions. We verified they are .jpg in images folder.
            filename = f'images/{img_id}.jpg'
            
            data_infos.append(dict(
                id=img_id,
                filename=filename,
                width=info['width'],
                height=info['height'],
                ann=info # pass raw info
            ))
        return data_infos

    def get_ann_info(self, idx):
        info = self.data_infos[idx]
        bbox_info_list = info['ann'].get('bbox', [])
        
        bboxes = []
        labels = []
        masks = []
        
        # Load global mask
        mask_path = osp.join(self.img_prefix, 'masks', f"{info['id']}.jpg")
        mask_all = None
        if osp.exists(mask_path):
             mask_all = mmcv.imread(mask_path, flag='grayscale')
             mask_all = (mask_all > 0).astype(np.uint8)

        for bbox_item in bbox_info_list:
            if bbox_item.get('label') != 'polyp':
                continue
                
            xmin = bbox_item['xmin']
            ymin = bbox_item['ymin']
            xmax = bbox_item['xmax']
            ymax = bbox_item['ymax']
            
            bboxes.append([xmin, ymin, xmax, ymax])
            labels.append(0)
            
            # Create instance mask
            if mask_all is not None:
                instance_mask = np.zeros_like(mask_all)
                # Crop global mask to bbox
                # Clamp coordinates
                h, w = mask_all.shape
                x1 = max(0, int(xmin))
                y1 = max(0, int(ymin))
                x2 = min(w, int(xmax))
                y2 = min(h, int(ymax))
                
                instance_mask[y1:y2, x1:x2] = mask_all[y1:y2, x1:x2]
                masks.append(instance_mask)
            
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
            masks = [] # Empty
        else:
            bboxes = np.array(bboxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
            
        # Create BitmapMasks
        if masks:
            masks = BitmapMasks(masks, info['height'], info['width'])
        else:
            # MMDetection often creates empty masks if list is empty but we need to match shape?
            # Actually, if bboxes is empty, masks should be empty list or BitmapMasks with 0 masks.
            # Pipeline handles empty masks usually.
             masks = BitmapMasks([], info['height'], info['width'])

        return dict(
            bboxes=bboxes,
            labels=labels,
            masks=masks,
            map_classes=np.array([0], dtype=np.int64)
        )

    def evaluate(self, results, metric='mAP', **kwargs):
        if isinstance(results[0], tuple):
             mask_results = [res[1] for res in results]
             results = [res[0] for res in results]
        else:
             # Should be tuple if running with Mask R-CNN and mask config
             mask_results = None

        eval_results = super().evaluate(results, metric, **kwargs)
        
        # Calculate Medical Metrics (Dice, IoU, Hausdorff)
        if mask_results is not None:
             print("\nCalculating Medical Metrics (Dice, IoU, Hausdorff)...")
             mean_dice = []
             mean_iou = []
             mean_hd = []
             
             try:
                from scipy.spatial import cKDTree
                has_scipy = True
             except ImportError:
                has_scipy = False
                print("Scipy not found. Hausdorff distance calculation will be skipped or approximated.")

             for i, masks in enumerate(mask_results):
                 # Get GT masks
                 ann = self.get_ann_info(i)
                 gt_masks_obj = ann.get('masks', None)
                 
                 if gt_masks_obj is None or len(gt_masks_obj.masks) == 0:
                     continue
                 
                 gt_masks = gt_masks_obj.to_ndarray() # [N_gt, H, W]
                 
                 # Masks are list of list (one list per class). Kvasir has 1 class.
                 # masks is list of len 1 (for class 0) -> containing list of [N_pred] masks (RLE or bool?)
                 # MMDetection mask results are typically RLE if encoded, or list of bool arrays
                 
                 # Unpack masks for class 0
                 pred_masks_cls0 = masks[0]
                 if len(pred_masks_cls0) == 0:
                      # No predictions
                      mean_dice.append(0.0)
                      mean_iou.append(0.0)
                      # HD is undefined or infinite for empty set
                      continue
                 
                 # We need to decode RLE if they are RLE. 
                 # However, MMDetection test output often keeps them as is.
                 # Let's assume we decode or they are decoded.
                 # Usually inference_detector outputs standard masks.
                 
                 # Combine all instance masks into one binary mask for semantic-style metric?
                 # OR match instances. 
                 # Medical segmentation often checks overlap of "polyp vs bg".
                 # So we collapse all instances to one mask.
                 
                 h, w = gt_masks.shape[1], gt_masks.shape[2]
                 pred_mask_all = np.zeros((h, w), dtype=np.uint8)
                 
                 import pycocotools.mask as mask_util
                 for m in pred_masks_cls0:
                     if isinstance(m, dict): # RLE
                         m = mask_util.decode(m)
                     pred_mask_all = np.maximum(pred_mask_all, m.astype(np.uint8))
                 
                 gt_mask_all = np.zeros((h, w), dtype=np.uint8)
                 for m in gt_masks:
                     gt_mask_all = np.maximum(gt_mask_all, m.astype(np.uint8))
                 
                 # Binary metrics
                 intersection = np.sum(pred_mask_all * gt_mask_all)
                 sum_pred = np.sum(pred_mask_all)
                 sum_gt = np.sum(gt_mask_all)
                 
                 dice = (2. * intersection) / (sum_pred + sum_gt + 1e-6)
                 iou = intersection / (sum_pred + sum_gt - intersection + 1e-6)
                 
                 mean_dice.append(dice)
                 mean_iou.append(iou)
                 
                 # Hausdorff 95%
                 if has_scipy and sum_pred > 0 and sum_gt > 0:
                      pred_pts = np.argwhere(pred_mask_all)
                      gt_pts = np.argwhere(gt_mask_all)
                      
                      # Use KDTree for efficient nearest neighbor search
                      tree_gt = cKDTree(gt_pts)
                      tree_pred = cKDTree(pred_pts)
                      
                      d_pred_to_gt, _ = tree_gt.query(pred_pts)
                      d_gt_to_pred, _ = tree_pred.query(gt_pts)
                      
                      hd95 = np.percentile(np.r_[d_pred_to_gt, d_gt_to_pred], 95)
                      mean_hd.append(hd95)
             
             if mean_dice:
                 eval_results['mDice'] = np.mean(mean_dice)
                 eval_results['mIoU'] = np.mean(mean_iou)
                 if has_scipy:
                     eval_results['mHD'] = np.mean(mean_hd) if mean_hd else 0.0

        return eval_results
