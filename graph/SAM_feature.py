import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from torch import nn

device = torch.device('cuda')
sam_checkpoint = "" # SAM weights
model_type = "vit_h"

class SAM_Feature(nn.Module):
    def __init__(self):
        super().__init__()
        sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
        for param in sam_model.parameters():
            param.requires_grad = False
        self.mask_predictor = SamPredictor(sam_model)

    def forward(self, image_metas):
        if self.training:
            images = [meta['img_flip'] for meta in image_metas]
        else:
            images = [np.array(meta['img_flip'][0]) for meta in image_metas]
        image_feature = []

        with torch.no_grad():
            for index, image in enumerate(images):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.mask_predictor.set_image(image)

                image_embedding = self.mask_predictor.get_image_embedding()
                image_feature.append(image_embedding)

        return torch.cat(image_feature, dim=0)
