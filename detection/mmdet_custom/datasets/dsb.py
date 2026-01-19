
import os
import os.path as osp
import json
import numpy as np
import mmcv
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmdet.core import BitmapMasks

@DATASETS.register_module()
class DSBDataset(CustomDataset):
    CLASSES = ('nuclei',)

    def load_annotations(self, ann_file):
        data_infos = []
        # img_prefix should point to 'stage1_train' directory
        ids = [d for d in os.listdir(self.img_prefix) if osp.isdir(osp.join(self.img_prefix, d))]
        
        for id_ in ids:
            img_rel_path = osp.join(id_, 'images', f'{id_}.png')
            abs_img_path = osp.join(self.img_prefix, img_rel_path)
            
            if not osp.exists(abs_img_path):
                continue
                
            # Read image to get shape
            # DSB images are small, so this is acceptable overhead for init
            img = mmcv.imread(abs_img_path)
            height, width = img.shape[:2]
            
            data_infos.append(dict(
                id=id_,
                filename=img_rel_path,
                width=width,
                height=height
            ))
        return data_infos

    def get_ann_info(self, idx):
        info = self.data_infos[idx]
        img_id = info['id']
        abs_path = osp.join(self.img_prefix, img_id)
        
        bboxes = []
        labels = []
        masks = []
        
        # Masks are in 'masks' subdir, multiple pngs
        mask_path = osp.join(abs_path, 'masks')
        if not osp.exists(mask_path):
             # Should not happen
             pass
        else:
            mask_files = [osp.join(mask_path, m) for m in os.listdir(mask_path) if m.endswith('.png')]
            for mf in mask_files:
                mask = mmcv.imread(mf, flag='grayscale')
                mask = (mask > 0).astype(np.uint8)
                
                # Bbox
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                if not np.any(rows) or not np.any(cols):
                    continue
                    
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]
                
                # Ensure valid bbox
                if (xmax - xmin) < 1 or (ymax - ymin) < 1:
                    continue

                bboxes.append([xmin, ymin, xmax, ymax])
                labels.append(0) 
                masks.append(mask)
            
        if not bboxes:
            # Empty placeholders if no objects found (should not happen in train)
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
            masks = []
        else:
            bboxes = np.array(bboxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
            
        return dict(
            bboxes=bboxes,
            labels=labels,
            masks=BitmapMasks(masks, info['height'], info['width']),
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
                 
                 # gt_masks: [N_gt, H, W]
                 gt_masks = gt_masks_obj.to_ndarray() 
                 h, w = gt_masks.shape[1], gt_masks.shape[2]

                 # Pred masks
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
