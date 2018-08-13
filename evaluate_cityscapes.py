import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys
from packaging import version

import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
from model.deeplab_multi import DeeplabMulti
from model.deeplab_vgg import DeeplabVGG
from dataset.cityscapes_dataset import cityscapesDataSet
from collections import OrderedDict
import os
import cv2
from PIL import Image

import matplotlib.pyplot as plt
import torch.nn as nn
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = './data/Cityscapes/data'
DATA_LIST_PATH = './dataset/cityscapes_list/val.txt'
SAVE_PATH = './result/cityscapes'

IGNORE_LABEL = 255
NUM_CLASSES = 19
NUM_STEPS = 500 # Number of images in the validation set.
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_multi-ed35151c.pth'
RESTORE_FROM_VGG = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_vgg-ac4ac9f6.pth'
SET = 'val'

MODEL = 'DeeplabMulti'

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def VideoCapturee():
    cap=cv2.VideoCapture(r'/media/avidbeam/Workspace/Abdelrahman-Ws/AdaptSegNet/data/Cityscapes/data/leftImg8bit/val/frankfurt/IMG_3198.MOV')

    cnt=0

    while(cap.isOpened()):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        np.rot90(frame)
        #frame=cv2.flip(frame,0)
        #cv2.transpose(frame,frame)
        #frame=cv2.flip(frame,frame, +1)
        #cv2.imshow('frame',gray)
        ss = str(cnt)
        sz = len(ss)
        kk = "00"
        for j in range(4 - sz):
            kk = kk + '0'

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.imwrite(r'/media/avidbeam/Workspace/Abdelrahman-Ws/AdaptSegNet/data/Cityscapes/data/leftImg8bit/val/frankfurt/' +kk + str(cnt)+".png",frame)
        cnt=cnt+1
        if(cnt>600):
            break
    cap.release()
    cv2.destroyAllWindows()



def Final_segmentation():

    Ground_Truth=os.listdir(r'/media/avidbeam/Workspace/Abdelrahman-Ws/AdaptSegNet/data/Cityscapes/data/leftImg8bit/val/frankfurt')

    Ground_Truth.sort()

    Colored=os.listdir(r'/media/avidbeam/Workspace/Abdelrahman-Ws/AdaptSegNet/result/cityscapes')

    Colored.sort()

    cnt=0

    for i in range(600):


        image_path = os.path.join(r'/media/avidbeam/Workspace/Abdelrahman-Ws/AdaptSegNet/data/Cityscapes/data/leftImg8bit/val/frankfurt', Ground_Truth[i])
        gt=cv2.imread(image_path)
        gt = cv2.resize(gt, (600, 600))
        gt=np.asanyarray(gt)


        image_path = os.path.join(r'/media/avidbeam/Workspace/Abdelrahman-Ws/AdaptSegNet/result/cityscapes',Colored[i])
        cld = cv2.imread(image_path)
        cld = cv2.resize(cld, (600, 600))
        #cv2.imshow("oyr", cld)
        cld = np.asanyarray(cld)
        Seg = np.zeros([600, 600, 3], dtype="uint8")

        for i in range(600):
            for j in range(600):
                if (cld[i][j][0] == 60 and cld[i][j][1] == 20 and cld[i][j][2] == 220):
                    Seg[i][j][0] = 30 + (gt[i][j][0] * 0.5)
                    Seg[i][j][1] = 10 + (gt[i][j][1] * 0.5)
                    Seg[i][j][2] = 110 + (gt[i][j][2] * 0.5)
                else:
                    Seg[i][j][0] = gt[i][j][0]
                    Seg[i][j][1] = gt[i][j][1]
                    Seg[i][j][2] = gt[i][j][2]
                    
        ss = str(cnt)
        sz = len(ss)
        kk = "00"
        for j in range(4 - sz):
            kk = kk + '0'
        cv2.imwrite("/media/avidbeam/Workspace/Abdelrahman-Ws/AdaptSegNet/ToBeWritten/" + kk + str(cnt) + ".png", Seg)
        cnt = cnt + 1



    
def VideoWriter():

    print("Entered")
    #ap = argparse.ArgumentParser()
    #ap.add_argument("-ext", "--extension", required=False, default='png', help="extension name. default is 'png'.")
    #ap.add_argument("-o", "--output", required=False, default='output.mp4', help="output video file")
    #args1 = vars(ap.parse_args())
    #args1 = ap.parse_args()
    #print("Here")
    ext = "png"
    output = "output.mp4"


    dir_path = r'/media/avidbeam/Workspace/Abdelrahman-Ws/AdaptSegNet/ToBeWritten'
    #ext = args1['extension']
    #output = args1['output']

    images = []
    for f in os.listdir(dir_path):
        if f.endswith(ext):
            images.append(f)


    images.sort()
    #print(images)

    image_path = os.path.join(dir_path, images[0])
    frame = cv2.imread(image_path)
    #cv2.imshow('video',frame)
    height, width, channels = frame.shape

# Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(output, fourcc, 15, (width, height))

    cnt=0
    for image in images:

        image_path = os.path.join(dir_path, image)
        frame = cv2.imread(image_path)

        out.write(frame)
        #cv2.imshow('video',frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break


    out.release()
    #cv2.destroyAllWindows()

    print("The output video is {}".format(output))



def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Model Choice (DeeplabMulti/DeeplabVGG).")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    return parser.parse_args()


def main():
    """Create the model and start the evaluation process."""

    

    args = get_arguments()

    gpu0 = args.gpu

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.model == 'DeeplabMulti':
        model = DeeplabMulti(num_classes=args.num_classes)
    elif args.model == 'DeeplabVGG':
        model = DeeplabVGG(num_classes=args.num_classes)
        if args.restore_from == RESTORE_FROM:
            args.restore_from = RESTORE_FROM_VGG

    if args.restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda(gpu0)

    testloader = data.DataLoader(cityscapesDataSet(args.data_dir, args.data_list, crop_size=(1024, 512), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
                                    batch_size=1, shuffle=False, pin_memory=True)


    if version.parse(torch.__version__) >= version.parse('0.4.0'):
        interp = nn.Upsample(size=(720, 1280), mode='bilinear', align_corners=True)
    else:
        interp = nn.Upsample(size=(720, 1280), mode='bilinear')

    for index, batch in enumerate(testloader):
        if index % 100 == 0:
            print '%d processd' % index
        image, _, name = batch
        if args.model == 'DeeplabMulti':
            output1, output2 = model(Variable(image, volatile=True).cuda(gpu0))
            output = interp(output2).cpu().data[0].numpy()
        elif args.model == 'DeeplabVGG':
            output = model(Variable(image, volatile=True).cuda(gpu0))
            output = interp(output).cpu().data[0].numpy()

        output = output.transpose(1,2,0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        output_col = colorize_mask(output)
        output = Image.fromarray(output)

        name = name[0].split('/')[-1]
        #output.save('%s/%s' % (args.save, name))
        output_col.save('%s/%s_color.png' % (args.save, name.split('.')[0]))



if __name__ == '__main__':
    VideoCapturee()
    main()
    Final_segmentation()
    VideoWriter()

