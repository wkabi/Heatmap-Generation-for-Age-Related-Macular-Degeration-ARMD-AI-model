
'''
Project: Age-Related Macular Degeneration (ARMD) classification using retinal fundus images into UNCLASS, AMD and NO-AMD
Author: Dr. Waziha Kabir
Code summary:
             ## Input: Pre-processed fundus images from a directory, the DR+AMD models, hyper-parameters
	     ## Output: Prediction of ARMD (incorporates DR(UNCLASS)=ARMD(UNCLASS))
'''

import argparse
import pandas as pd
from models.get_model import get_arch
from utils.get_loaders import get_test_from_folder_loader, get_test_cls_loader

from utils.reproducibility import set_seeds
from utils.model_saving_loading import load_model
from tqdm import trange
import numpy as np
import torch
import torchvision

import os.path as osp
import os
import sys
from torchvision import models
from PIL import Image
from torchvision import transforms as tr
import torch.nn.functional as F

## Additional libraries for heatmaps [START]
from pytorch_grad_cam import GradCAM, HiResCAM, GradCAMElementWise, GradCAMPlusPlus, XGradCAM, AblationCAM, ScoreCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad
from PIL import Image
import torchvision.transforms as T
from skimage import io
import glob
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import shutil
## Additional libraries for heatmaps [END]

def str2bool(v):
    # as seen here: https://stackoverflow.com/a/43357954/3208255
    if isinstance(v, bool):
       return v
    if v.lower() in ('true','yes'):
        return True
    elif v.lower() in ('false','no'):
        return False
    else:
        raise argparse.ArgumentTypeError('boolean value expected.')

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type=str, default='verifications/input/images/', help='path test data')
parser.add_argument('--model_name', type=str, default='bit_resnext101_1', help='selected architecture')
parser.add_argument('--load_path', type=str, default='./DRMdl', help='path to saved DR model')
parser.add_argument('--dihedral_tta', type=int, default=0, help='dihedral group cardinality (0)')
parser.add_argument('--im_size', help='delimited list input, could be 500, or 600,400', type=str, default='512,512')
parser.add_argument('--n_classes', type=int, default=6, help='number of DR target classes (6)')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--results_path', type=str, default='./verifications/output/', help='path to output csv')
parser.add_argument('--csv_out', type=str, default='results.csv', help='path to output csv')
parser.add_argument('--csv_path_test', type=str, default='verifications/input/images.csv', help='path to input csv with image_id')

## ARMD arguments
parser.add_argument('--data_path_AMD', type=str, default='verifications/input/cropped_images/', help='path test data')
parser.add_argument('--model_name_AMD', type=str, default='resnext50_tv', help='selected architecture')
parser.add_argument('--load_path_AMD', type=str, default='./', help='path to saved model')
parser.add_argument('--dihedral_tta_AMD', type=int, default=1, help='dihedral group cardinality (0)')
parser.add_argument('--n_classes_AMD', type=int, default=1, help='number of target classes (1)')
## ARMD arguments

### Additional arguments for HeatMaps
parser.add_argument('--save_dir_NT', default='resultsHMNT',type=str, help='path to save results')
parser.add_argument('--img_ext', default='jpg', type=str, help='type of image extension')
parser.add_argument('--csv_path', default='./verifications/output/output.csv', type=str, help='csv path to save probability')
parser.add_argument('--save_dir', default='./verifications/output/resultsHM',type=str, help='path to save results')
parser.add_argument('--save_dir_RawHM', default='./verifications/output/resultsRawHM',type=str, help='path to save results')

args = parser.parse_args()

def run_one_epoch_cls(loader, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    probs_all, preds_all, labels_all = [], [], []
    with trange(len(loader)) as t:
        for i_batch, inputs in enumerate(loader):
            inputs = inputs.to(device, non_blocking=True)
            logits = model(inputs)
            probs = torch.nn.Softmax(dim=1)(logits)
            _, preds = torch.max(probs, 1)
            probs_all.extend(probs.detach().cpu().numpy())
            preds_all.extend(preds.detach().cpu().numpy())
            run_loss = 0
            t.set_postfix(vl_loss="{:.4f}".format(float(run_loss)))
            t.update()
    return np.stack(preds_all), np.stack(probs_all)

def test_cls_tta_dihedral(model, test_loader, n=1):
    probs_tta = []
    prs = [0, 1]

    test_loader.dataset.transforms.transforms.insert(-1, torchvision.transforms.RandomRotation(0))
    rotations = np.array([i * 360 // n for i in range(n)])
    for angle in rotations:
        for p2 in prs:
            test_loader.dataset.transforms.transforms[2].p = p2  # pr(vertical flip)
            test_loader.dataset.transforms.transforms[-2].degrees = [angle, angle]
            with torch.no_grad():
                test_preds, test_probs = run_one_epoch_cls(test_loader, model)
                probs_tta.append(test_probs.squeeze())

    probs_tta = np.mean(np.array(probs_tta), axis=0)
    preds_tta = np.argmax(probs_tta, axis=1)

    del model
    torch.cuda.empty_cache()
    return probs_tta, preds_tta

def test_cls(model, test_loader):
    # validate one epoch, note no optimizer is passed
    with torch.no_grad():
        test_preds, test_probs = run_one_epoch_cls(test_loader, model)

    del model
    torch.cuda.empty_cache()
    return test_probs, test_preds

## [START] WAZIHA modifications: Function that grades DR probablities using Diagnos production thresholding (Scenario July 2023)
## [START] WAZIHA modifications: Function that grades ARMD probablities using Diagnos production thresholding (Scenario September 2023)
def scenario_3_DR_AMD(list_of_DR_preds, list_of_AMD_preds):
  Grade_DR_all = []
  Grade_AMD_all = []
  for i in range(len(list_of_DR_preds)):
        # DR Grades assignment for UNCLASS
        if (list_of_DR_preds[i,5] >= 0.24210012):
                Grade_DR = 'UNCLASS'
        else:
                Grade_DR = 'CLASS'
        Grade_DR_all.append(Grade_DR)

	# ARMD Grades assignment according to DR Grading for 'UNCLASS'(September 2023)
        if (Grade_DR == 'UNCLASS'):
                Grade_AMD = 'UNCLASS'
        elif (list_of_AMD_preds[i] == True):
                Grade_AMD = 'AMD'
        else:
                Grade_AMD = 'NO_AMD'
        Grade_AMD_all.append(Grade_AMD)

  return Grade_DR_all, Grade_AMD_all
## [END] WAZIHA modifications: Function that grades DR probablities using Diagnos production thresholding (Scenario July 2023)
## [END] WAZIHA modifications: Function that grades ARMD probablities using Diagnos production thresholding (Scenario September 2023)

def run_one_epoch_cls_AMD(loader, model, optimizer=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train = optimizer is not None
    model.train() if train else model.eval()
    probs_all, preds_all, labels_all = [], [], []
    with trange(len(loader)) as t:
        for i_batch, inputs in enumerate(loader):
            if loader.dataset.has_labels:
                (inputs, labels, _) = inputs
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            else:
                inputs = inputs.to(device, non_blocking=True)
            logits = model(inputs)
            probs = logits.sigmoid()

            probs = probs.detach().cpu().numpy().squeeze()
            if probs.ndim == 0: # handles case of 1-item batch
                probs = np.expand_dims(probs, axis=0)
            preds = probs > 0.14363253 ## Waziha-Jihed threshold

            probs_all.extend(probs)
            preds_all.extend(preds)

            if loader.dataset.has_labels:
                labels_all.extend(labels.detach().cpu().numpy())
            run_loss = 0
            t.set_postfix(vl_loss="{:.4f}".format(float(run_loss)))
            t.update()
    if loader.dataset.has_labels:
        return np.stack(preds_all), np.stack(probs_all), np.stack(labels_all)
    return np.stack(preds_all), np.stack(probs_all), None

def test_cls_tta_dihedral_AMD(model, test_loader, n=3):
    probs_tta = []
    prs = [0, 1]

    test_loader.dataset.transforms.transforms.insert(-1, torchvision.transforms.RandomRotation(0))
    rotations = np.array([i * 360 // n for i in range(n)])
    for angle in rotations:
        for p2 in prs:
            test_loader.dataset.transforms.transforms[2].p = p2  # pr(vertical flip)
            test_loader.dataset.transforms.transforms[-2].degrees = [angle, angle]
            # validate one epoch, note no optimizer is passed
            with torch.no_grad():
                test_preds, test_probs, test_labels = run_one_epoch_cls_AMD(test_loader, model)
                probs_tta.append(test_probs)

    probs_tta = np.mean(np.array(probs_tta), axis=0)
    preds_tta = probs_tta > 0.14363253 ## Waziha-Jihed threshold

    del model
    torch.cuda.empty_cache()
    return probs_tta, preds_tta, test_labels

def test_cls_AMD(model, test_loader):
    # validate one epoch, note no optimizer is passed
    with torch.no_grad():
        test_preds, test_probs, test_labels = run_one_epoch_cls_AMD(test_loader, model)

    del model
    torch.cuda.empty_cache()
    return test_probs, test_preds, test_labels

tfm = torch.nn.Sequential(
    T.Resize(size=(512,512)))
    
def process_img(img_path, save_dir, model, save_dir_RawHM):

    fname = os.path.basename(img_path)
    
    image = io.imread(img_path,as_gray=True)
    image = image / np.max(image)

    inp = torch.from_numpy(image).float()
    inp = inp.unsqueeze(0).unsqueeze(0)
 
    #tfm = torch.nn.Sequential(T.Resize(size=(512,512)))

    inp = tfm(inp)[0][0]
    inp = torch.stack([inp, inp, inp],dim=0)

    inp = inp.unsqueeze(0).to(device)
    
    #outputs = model(inp)
    #predicts = torch.sigmoid(outputs).cpu().detach().numpy()
    #pred = np.argmax(predicts,axis=1)[0]
    #conf = predicts[:,0][0]
    
    #target_layers = [model.layer2[-1]] #Waziha's selection Layer 2 for Normal/Abnormal
    #target_layers = [model.layer3[-1]] #Waziha's selection Layer 3 for Normal/Abnormal
    target_layers = [model.layer4[-1]] #Waziha's selection Layer 4 for Normal/Abnormal
   
    #cam = GradCAM(model = model, target_layers=target_layers, use_cuda=True)
    #cam = GradCAM(model = model, target_layers=target_layers)
    #cam = HiResCAM(model = model, target_layers=target_layers)
    #cam = GradCAMElementWise(model = model, target_layers=target_layers)
    #cam = GradCAMPlusPlus(model = model, target_layers=target_layers)
    #cam = XGradCAM(model = model, target_layers=target_layers)
    #cam = AblationCAM(model = model, target_layers=target_layers)
    #cam = ScoreCAM(model = model, target_layers=target_layers)
    cam = EigenCAM(model = model, target_layers=target_layers)
    #cam = EigenGradCAM(model = model, target_layers=target_layers)
    #cam = LayerCAM(model = model, target_layers=target_layers)
    #cam = FullGrad(model = model, target_layers=target_layers)

    grayscale_cam = cam(input_tensor=inp)
    grayscale_cam = grayscale_cam[0, :]
    grayscale_cam = cam(input_tensor=inp)
    grayscale_cam = grayscale_cam[0, :]
    plt.imshow(grayscale_cam, cmap='jet')
    plt.axis('off')
    plt.savefig('{}/{}.RawHM.png'.format(save_dir_RawHM,fname[:-4]),bbox_inches='tight')

    rgb_img = inp[0].cpu().numpy().transpose(1,2,0)
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    #print(visualization.size)
    plt.imshow(visualization)
    plt.axis('off')
    plt.savefig('{}/{}.HM.png'.format(save_dir,fname[:-4]),bbox_inches='tight')

    return
    
if __name__ == '__main__':
    '''
    Example:

    '''
    data_path = args.data_path
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # reproducibility
    seed_value = 0
    set_seeds(seed_value, use_cuda)

    # gather parser parameters
    args = parser.parse_args()
    model_name = args.model_name
    load_path = args.load_path
    results_path = osp.join(args.results_path)
    os.makedirs(results_path, exist_ok=True)
    bs = args.batch_size
    n_classes = args.n_classes
    im_size = tuple([int(item) for item in args.im_size.split(',')])
    if isinstance(im_size, tuple) and len(im_size)==1:
        tg_size = (im_size[0], im_size[0])
    elif isinstance(im_size, tuple) and len(im_size)==2:
        tg_size = (im_size[0], im_size[1])
    else:
        sys.exit('im_size should be a number or a tuple of two numbers')
    dihedral_tta = args.dihedral_tta

    ## [START] Compute DR probabilities
    model, mean, std = get_arch(model_name, n_classes=n_classes)
    model, stats = load_model(model, load_path, device='cpu')
    model = model.to(device)

    csv_path_test = args.csv_path_test
    #test_loader = get_test_from_folder_loader(data_path=data_path,  batch_size=bs, mean=mean, std=std, tg_size=tg_size)
    test_loader = get_test_cls_loader(csv_path_test=csv_path_test, data_path=data_path, batch_size=bs, tg_size=tg_size, mean=mean, std=std, test=True)

    if dihedral_tta==0:
        probs, preds = test_cls(model, test_loader)
    elif dihedral_tta>0:
        probs, preds = test_cls_tta_dihedral(model, test_loader, n=dihedral_tta)
    else: sys.exit('dihedral_tta must be >=0')
    ## [END] Compute DR probabilities

    ## [START] Compute ARMD probabilities
    # gather parser parameters
    data_path_AMD = args.data_path_AMD
    model_name_AMD = args.model_name_AMD
    load_path_AMD = args.load_path_AMD
    #results_path = osp.join(args.results_path, load_path_AMD.split('/')[1], load_path_AMD.split('/')[2])
    results_path = args.results_path
    os.makedirs(results_path, exist_ok=True)
    n_classes_AMD = args.n_classes_AMD
    im_size = tuple([int(item) for item in args.im_size.split(',')])
    if isinstance(im_size, tuple) and len(im_size)==1:
        tg_size = (im_size[0], im_size[0])
    elif isinstance(im_size, tuple) and len(im_size)==2:
        tg_size = (im_size[0], im_size[1])
    else:
        sys.exit('im_size should be a number or a tuple of two numbers')
    dihedral_tta_AMD = args.dihedral_tta_AMD

    print('* Loading AMD model {} from {}'.format(model_name_AMD, load_path_AMD))
    model_AMD, mean_AMD, std_AMD = get_arch(model_name_AMD, n_classes=n_classes_AMD)
    model_AMD, stats_AMD = load_model(model_AMD, load_path_AMD, device='cpu')
    model_AMD = model_AMD.to(device)
    print("Total params: {0:,}".format(sum(p.numel() for p in model_AMD.parameters() if p.requires_grad)))

    print('* Creating Test Dataloaders, batch size = {:d}'.format(bs))
    test_loader_AMD = get_test_cls_loader(data_path=data_path_AMD, csv_path_test=csv_path_test,  batch_size=bs, mean=mean_AMD, std=std_AMD, tg_size=tg_size, test=True)

    if dihedral_tta_AMD==0:
        probs_AMD, preds_AMD, label_AMD = test_cls_AMD(model_AMD, test_loader_AMD)
    elif dihedral_tta_AMD>0:
        probs_AMD, preds_AMD, label_AMD = test_cls_tta_dihedral_AMD(model_AMD, test_loader_AMD, n=dihedral_tta_AMD)
    else: sys.exit('dihedral_tta_AMD must be >=0')
    ## [END] Compute ARMD probabilities

    ## [START] Grading of DR & AMD probablities (Scenario September 2023)
    gradeDR_sc2, gradeAMD_sc2 = scenario_3_DR_AMD(probs, preds_AMD)
    ## [END] Grading of DR & AMD probablities (Scenario September 2023)

    ## [START] Saving grading of DR and ARMD into .csv file
    if n_classes==6:
        im_list = list(test_loader.dataset.im_list)
        df_nogt = pd.DataFrame(zip(im_list, probs[:, 5], probs_AMD, gradeAMD_sc2),
                          columns=['image_id','u', 'AMD', 'AMD_GRADE_sc2'])
    else: sys.exit('Wrong number of classes when saving dataframe')

    df_nogt.to_csv(osp.join(results_path, args.csv_out), index=False)
    ## [END] Saving grading of DR & ARMD into .csv file

## HeatMap generation [START]
    img_ext = args.img_ext
    img_dir = data_path_AMD
    save_dir = args.save_dir
    save_dir_NT = args.save_dir_NT
    csv_path = args.csv_path
    save_dir_RawHM = args.save_dir_RawHM

    if not os.path.exists(args.save_dir_NT):
        os.mkdir(args.save_dir_NT)

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    if not os.path.exists(args.save_dir_RawHM):
        os.mkdir(args.save_dir_RawHM)

    for img_path in glob.glob(img_dir + '/*.{}'.format(args.img_ext)):
        process_img(img_path, save_dir_NT, model_AMD, save_dir_RawHM)
        #conf = process_img(img_path, save_dir_NT, model)
## HeatMap generation [END]

    ## [START] WAZIHA modifications: Printing ARMD Grades for each images
    for i in range(len(im_list)):
       print("DATA_TAG:=AMD={};".format(gradeAMD_sc2[i]),"File={}".format(im_list[i]))
    ## [END] WAZIHA modifications: Printing ARMD Grades for each images
    
## Printing probability over heatmaps & printing other outputs [START]
    for r in range(len(im_list)):
        img_HM = Image.open(osp.join(save_dir_NT, (im_list[r].replace('.jpg','.HM.png'))))
        image_org = Image.open(osp.join(data_path_AMD, im_list[r]))
        new_width, new_height = image_org.size
        #print(image_org.size)
        #print(img_HM.size)
        img_HM_rz = img_HM.resize((new_width, new_height))
        plt.title(probs_AMD[r])
        plt.imshow(img_HM_rz)
        plt.axis('off') 
        plt.savefig('{}/{}'.format(save_dir,im_list[r].replace('.jpg','.HM_rz.png')), bbox_inches='tight', facecolor='w', transparent=True)
## Printing probability over heatmaps & printing other outputs [END]
#shutil.rmtree(save_dir_NT)    
