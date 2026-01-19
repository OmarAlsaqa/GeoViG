from mmdet.datasets import PIPELINES

@PIPELINES.register_module()
class LoadMasksFromAnn:
    """Load masks directly from ann_info (e.g. if likely already BitmapMasks)."""
    def __call__(self, results):
        if 'masks' in results['ann_info']:
            results['gt_masks'] = results['ann_info']['masks']
            results['mask_fields'].append('gt_masks')
        return results
