from PIL import Image
from faster_rcnn.frcnn import FRCNN
from retinanet.retinanet import Retinanet

model_name = "retinanet"

if model_name == "frcnn":
    model_path = '/home/hjh/PJ2/logs_frcnn/best_epoch_weights.pth'
    frcnn = FRCNN(model_path = model_path,confidence = 0.6, nms_iou = 0.3, base_sizes = [8, 16, 32])

    stage = 1
    path=[]
    path.append('/home/hjh/PJ2/VOC2007/JPEGImages/005808.jpg')
    path.append('/home/hjh/PJ2/VOC2007/JPEGImages/005036.jpg')
    path.append( '/home/hjh/PJ2/VOC2007/JPEGImages/007447.jpg')
    path.append('/home/hjh/PJ2/VOC2007/JPEGImages/003324.jpg')

    for i in range(4):
        image = Image.open(path[i])
        r_image = frcnn.image_rst(image, stage=stage, rpn_num=20)
        r_image.save('/home/hjh/PJ2/example_img/test'+str(i)+'_rpn.jpg')

    stage = 2
    for i in range(1,4):
        path = '/home/hjh/PJ2/example_img/example'
        name = str(i)+'.jpg'
        try:
            image = Image.open(path+name)
        except:
            print('Open Error! Try again!')
        else:
            r_image = frcnn.image_rst(image,stage=stage)
            r_image.save(path+str(i)+'_frcnn.jpg')

elif model_name == "retinanet":
    model_path = '/home/hjh/PJ2/logs_retinanet/best_epoch_weights.pth'
    retinanet = Retinanet(model_path = model_path,confidence = 0.6, nms_iou = 0.3, base_sizes = [8, 16, 32])
    for i in range(1,4):
        path = '/home/hjh/PJ2/example_img/example'
        name = str(i)+'.jpg'
        try:
            image = Image.open(path+name)
        except:
            print('Open Error! Try again!')
        else:
            r_image = retinanet.image_rst(image)
            r_image.save(path+str(i)+'_retinanet.jpg')
else:
    print("Wrong model!")

        

