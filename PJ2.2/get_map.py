import os
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm
from utils.utils import get_classes
from utils.utils_map import get_map
from faster_rcnn.frcnn import FRCNN
from retinanet.retinanet import Retinanet

if __name__ == "__main__":

    model_name = "retinanet"
    
    classes_path    = '/home/hjh/PJ2/faster_rcnn/model_data/voc_classes.txt'
    MINOVERLAP      = 0.5
    confidence      = 0.02
    nms_iou         = 0.5
    score_threhold  = 0.5
    VOCdevkit_path  = '/home/hjh/PJ2'
    map_out_path    = 'map_out'

    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt")).read().strip().split()

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    if model_name == 'frcnn':
        model_path = '/home/hjh/PJ2/logs_frcnn/best_epoch_weights.pth'
        frcnn = FRCNN(model_path =model_path, confidence = confidence, nms_iou = nms_iou)
    elif model_name == 'retinanet':
        model_path = '/home/hjh/PJ2/logs_retinanet/best_epoch_weights.pth'
        retinanet = Retinanet(model_path =model_path, confidence = confidence, nms_iou = nms_iou)
    else:
        print('Wrong model!')

    class_names, _ = get_classes(classes_path)

    print("Get predict result.")
    for image_id in tqdm(image_ids):
        image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
        image       = Image.open(image_path)
        if model_name == 'frcnn':
            frcnn.get_map_txt(image_id, image, class_names, map_out_path)
        elif model_name == 'retinanet':
            retinanet.get_map_txt(image_id, image, class_names, map_out_path)
        else:
            print('Wrong model!')
    print("Get predict result done.")
    
    print("Get ground truth result.")
    for image_id in tqdm(image_ids):
        with open(os.path.join(map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
            root = ET.parse(os.path.join(VOCdevkit_path, "VOC2007/Annotations/"+image_id+".xml")).getroot()
            for obj in root.findall('object'):
                difficult_flag = False
                if obj.find('difficult')!=None:
                    difficult = obj.find('difficult').text
                    if int(difficult)==1:
                        difficult_flag = True
                obj_name = obj.find('name').text
                if obj_name not in class_names:
                    continue
                bndbox  = obj.find('bndbox')
                left    = bndbox.find('xmin').text
                top     = bndbox.find('ymin').text
                right   = bndbox.find('xmax').text
                bottom  = bndbox.find('ymax').text

                if difficult_flag:
                    new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                else:
                    new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
    print("Get ground truth result done.")

    print("Get map.")
    get_map(MINOVERLAP, True, score_threhold = score_threhold, path = map_out_path)
    print("Get map done.")


