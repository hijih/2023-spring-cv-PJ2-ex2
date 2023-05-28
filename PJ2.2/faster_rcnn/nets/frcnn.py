import torch.nn as nn
import torchvision.models as models
from faster_rcnn.nets.classifier import VGG16RoIHead
from faster_rcnn.nets.rpn import RegionProposalNetwork

class FasterRCNN(nn.Module):
    def __init__(self,  class_num,  mode = "training",stride = 16,
                base_sizes = [8, 16, 32], ratios = [0.5, 1, 2]):
        super(FasterRCNN, self).__init__()
        vgg16 = models.vgg16()
        self.stride = stride
        self.extractor = nn.Sequential(*list(vgg16.features.children())[:-1])

        self.rpn = RegionProposalNetwork(ratios = ratios, stride = self.stride, mode  = mode, base_sizes = base_sizes)
        
        classifier = list(vgg16.classifier)
        del classifier[6]
        del classifier[5]
        del classifier[2]
        classifier = nn.Sequential(*classifier)
        
        self.head = VGG16RoIHead(
            n_class         = class_num + 1,
            roi_size        = 7,
            spatial_scale   = 1,
            classifier = classifier)
            
    def forward(self, x, scale=1., mode="forward"):
        if mode == "forward":

            img_size        = x.shape[2:]
            base_feature    = self.extractor.forward(x)
            _, _, rois, roi_indices, _  = self.rpn.forward(base_feature, img_size, scale)
            roi_cls_locs, roi_scores    = self.head.forward(base_feature, rois, roi_indices, img_size)
            return roi_cls_locs, roi_scores, rois, roi_indices
        elif mode == "extractor":
            base_feature    = self.extractor.forward(x)
            return base_feature
        elif mode == "rpn":
            base_feature, img_size = x 
            rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn.forward(base_feature, img_size, scale)
            return rpn_locs, rpn_scores, rois, roi_indices, anchor
        elif mode == "head":
            base_feature, rois, roi_indices, img_size = x
            roi_cls_locs, roi_scores    = self.head.forward(base_feature, rois, roi_indices, img_size)
            return roi_cls_locs, roi_scores

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
            