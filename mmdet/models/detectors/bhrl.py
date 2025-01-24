import os

import torch
import torch.nn as nn
from ..builder import DETECTORS
from .two_stage import TwoStageDetector
from ..plugins.match_module import MatchModule
from ..plugins.generate_ref_roi_feats import generate_ref_roi_feats

from graph.SIGNN import before_RPN_samfuse
from graph.SAM_feature import SAM_Feature


@DETECTORS.register_module()
class BHRL(TwoStageDetector):
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(BHRL, self).__init__(backbone=backbone,
                                       neck=neck,
                                       rpn_head=rpn_head,
                                       roi_head=roi_head,
                                       train_cfg=train_cfg,
                                       test_cfg=test_cfg,
                                       pretrained=pretrained,
                                       init_cfg=init_cfg)

        self.matching_block = MatchModule(512, 384)
        self.before_RPN_sam = before_RPN_samfuse(iter=1)
        self.sam_feature = SAM_Feature()

    def matching(self, img_feat, rf_feat):
        out = []
        for i in range(len(rf_feat)):
            out.append(self.matching_block(img_feat[i], rf_feat[i]))
        return out

    def extract_feat(self, img, img_metas):
        img_feat = img[0]
        rf_feat = img[1]
        rf_bbox = img[2]
        img_feat = self.backbone(img_feat)
        rf_feat = self.backbone(rf_feat)
        if self.with_neck:
            img_feat = self.neck(img_feat)
            rf_feat = self.neck(rf_feat)

        img_feat_metric = self.matching(img_feat, rf_feat)

        sam_feature = self.sam_feature(img_metas)
        img_feat_metric, img_feat, rf_feat = self.before_RPN_sam(img_feat, rf_feat, img_feat_metric, sam_feature)

        ref_roi_feats = generate_ref_roi_feats(rf_feat, rf_bbox)
        return tuple(img_feat_metric), tuple(img_feat), ref_roi_feats

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        x, img_feat, ref_roi_feats  = self.extract_feat(img, img_metas)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        result = self.roi_head.simple_test(
            x, img_feat, ref_roi_feats, proposal_list, img_metas, rescale=rescale)

        return result


