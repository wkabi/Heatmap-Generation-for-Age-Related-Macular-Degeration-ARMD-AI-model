import sys, os
import numpy as np
import torch
from . import bit_models, bit_models_MOD
import timm
from torchvision.models import resnet, mobilenet_v2, alexnet, squeezenet
from .repvgg import repvgg_model_convert, create_RepVGG_A0, create_RepVGG_A1, create_RepVGG_B1g4

def cum_derivative_left(tens_1d):
    p_cum = torch.nn.functional.pad(tens_1d, pad=(1,0))
    return (p_cum-p_cum.roll(shifts=1))[:, 1:]

# IMAGENET PERFORMANCE FOR TIMM MODELS IS HERE:
# https://github.com/rwightman/pytorch-image-models/blob/master/results/results-imagenet.csv
def get_arch(model_name, in_c=3, n_classes=1):

    if model_name == 'resnet18':
        model = resnet.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'resnet34':
        model = resnet.resnet34(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'squeezenet':
        model = squeezenet.squeezenet1_0(pretrained=True)
        model.classifier[1] = torch.nn.Conv2d(512, n_classes, kernel_size=(1,1), stride=(1,1))
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'resnet50':
        model = resnet.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'resnext50_tv':
        model = resnet.resnext50_32x4d(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'mobilenetV2':
        model = mobilenet_v2(pretrained=True)
        num_ftrs = model.classifier[1].in_features
        model.classifier = torch.nn.Linear(num_ftrs, n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'repvgg_A0': # comparable to resnet18
        model = create_RepVGG_A0(deploy=False)
        if not os.path.isfile('models/RepVGG-A0-train.pth'):
            print('downloading repvgg_A0 weights:')
            os.system('wget --no-check-certificate \'https://docs.google.com/uc?export=download&id=13Gn8rq1PztoMEgK7rCOPMUYHjGzk-w11\' -O models/RepVGG-A0-train.pth')
        model.load_state_dict(torch.load('models/RepVGG-A0-train.pth'))
        n_features = model.linear.in_features
        model.linear = torch.nn.Linear(n_features, n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'repvgg_A1': # comparable to resnet34
        model = create_RepVGG_A1(deploy=False)
        if not os.path.isfile('models/RepVGG-A1-train.pth'):
            print('downloading repvgg_A1 weights:')
            os.system('wget --no-check-certificate \'https://docs.google.com/uc?export=download&id=19lX6lNKSwiO5STCvvu2xRTKcPsSfWAO1\' -O models/RepVGG-A1-train.pth')
        model.load_state_dict(torch.load('models/RepVGG-A1-train.pth'))
        n_features = model.linear.in_features
        model.linear = torch.nn.Linear(n_features, n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'bit_resnext50_1':
        bit_variant = 'BiT-M-R50x1'
        model = bit_models.KNOWN_MODELS[bit_variant](head_size=n_classes, zero_head=True)
        if not os.path.isfile('models/BiT-M-R50x1.npz'):
            print('downloading bit_resnext50_1 weights:')
            os.system('wget https://storage.googleapis.com/bit_models/BiT-M-R50x1.npz -P models/')
        model.load_from(np.load('models/BiT-M-R50x1.npz'))
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    elif model_name == 'bit_resnext50_1_KD':
        bit_variant = 'BiT-M-R50x1'
        model = bit_models.KNOWN_MODELS[bit_variant](head_size=n_classes, zero_head=True)
        if not os.path.isfile('models/R50x1_224.npz'):
            print('downloading bit_resnext50_1 weights:')
            os.system('wget https://storage.googleapis.com/bit_models/distill/R50x1_224.npz -P models/')
        model.load_from(np.load('models/R50x1_224.npz'))
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    elif model_name == 'swin':
        model = timm.create_model('swin_base_patch4_window12_384', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'cait':  # 68.37M params	384x384
        model = timm.create_model('cait_s36_384', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


    elif model_name == 'vit_tiny_p16':
        model = timm.create_model('vit_tiny_patch16_384', pretrained=True, num_classes=n_classes)
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    elif model_name == 'vit_small_p16':
        model = timm.create_model('vit_small_patch16_384', pretrained=True, num_classes=n_classes)
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    elif model_name == 'vit_small_p32':
        model = timm.create_model('vit_small_patch32_384', pretrained=True, num_classes=n_classes)
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    elif model_name == 'vit_base_p16':
        model = timm.create_model('vit_base_patch16_384', pretrained=True, num_classes=n_classes)
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    elif model_name == 'vit_base_p32':
        model = timm.create_model('vit_base_patch32_384', pretrained=True, num_classes=n_classes)
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)



    elif model_name == 'efficientnet_b5': # 30.39M params	456x456
        model = timm.create_model('tf_efficientnet_b5_ns', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'efficientnet_b6': # 43.049M params	528x528
        model = timm.create_model('tf_efficientnet_b6_ns', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'efficientnet_b7':  # 66.35M params	600x600
        model = timm.create_model('tf_efficientnet_b7_ns', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'vit':
        model = timm.create_model('vit_large_patch16_384', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'cspresnet50':
        model = timm.create_model('cspresnet50', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'cspresnext50':
        model = timm.create_model('cspresnext50', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'dpn68b':
        model = timm.create_model('dpn68b', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'regnetx_002':
        model = timm.create_model('regnetx_002', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'regnety_002':
        model = timm.create_model('regnety_002', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'regnetx_004':
        model = timm.create_model('regnetx_004', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'regnety_004':
        model = timm.create_model('regnety_004', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'regnetx_016':
        model = timm.create_model('regnetx_016', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'regnety_016':
        model = timm.create_model('regnety_016', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'regnetx_032':
        model = timm.create_model('regnetx_032', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'regnety_032':
        model = timm.create_model('regnety_032', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'dla60_res2net':
        model = timm.create_model('dla60_res2net', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'dla60_res2next':
        model = timm.create_model('dla60_res2next', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'seresnext50_32x4d':
        model = timm.create_model('seresnext50_32x4d', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'ecaresnet50t':
        model = timm.create_model('ecaresnet50t', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'repvgg_a2':
        model = timm.create_model('repvgg_a2', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'repvgg_b1':
        model = timm.create_model('repvgg_b1', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'repvgg_b1g4':
        model = timm.create_model('repvgg_b1g4', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'repvgg_b2':
        model = timm.create_model('repvgg_b2', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'repvgg_b2g4':
        model = timm.create_model('repvgg_b2g4', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'repvgg_b3':
        model = timm.create_model('repvgg_b3', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'repvgg_b3g4':
        model = timm.create_model('repvgg_b3g4', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'res2net50_48w_2s':
        model = timm.create_model('res2net50_48w_2s', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'res2net50_14w_8s':
        model = timm.create_model('res2net50_14w_8s', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'res2net50_26w_6s':
        model = timm.create_model('res2net50_26w_6s', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'res2net50_26w_8s':
        model = timm.create_model('res2net50_26w_8s', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'dm_nfnet_f4':
        model = timm.create_model('dm_nfnet_f4', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'resnest50d':
        model = timm.create_model('resnest50d', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'resnest269e':
        model = timm.create_model('resnest269e', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'mobilenetv3_large_100':
        model = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'skresnext50_32x4d':
        model = timm.create_model('skresnext50_32x4d', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'selecsls42b':
        model = timm.create_model('selecsls42b', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'hrnet_w18':
        model = timm.create_model('hrnet_w18', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'bit_timm':
        model = timm.create_model('resnetv2_50x1_bitm_in21k', pretrained=True, num_classes=n_classes)
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    elif model_name == 'swsl_resnet50': #Waziha
        model = timm.create_model('swsl_resnet50', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'cspresnext50': #Waziha
        model = timm.create_model('cspresnext50', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'ens_adv_inception_resnet_v2': #Waziha
        model = timm.create_model('ens_adv_inception_resnet_v2', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'gluon_resnext101_32x4d': #Waziha
        model = timm.create_model('gluon_resnext101_32x4d', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'gluon_seresnext101_32x4d': #Waziha
        model = timm.create_model('gluon_seresnext101_32x4d', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'inception_v4': #Waziha
        model = timm.create_model('inception_v4', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'rexnet_100': #Waziha
        model = timm.create_model('rexnet_100', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'resnet101d': #Waziha
        model = timm.create_model('resnet101d', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'tf_efficientnet_b0_ns': #Waziha
        model = timm.create_model('tf_efficientnet_b0_ns', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'mixnet_l': #Waziha
        model = timm.create_model('mixnet_l', pretrained=True, num_classes=n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif model_name == 'bit_resnext50_1_MOD':
        bit_variant = 'BiT-M-R50x1'
        model = bit_models_MOD.KNOWN_MODELS[bit_variant](head_size=n_classes, zero_head=True)
        if not os.path.isfile('models/BiT-M-R50x1.npz'):
            print('downloading bit_resnext50_1 weights:')
            os.system('wget https://storage.googleapis.com/bit_models/BiT-M-R50x1.npz -P models/')
        model.load_from(np.load('models/BiT-M-R50x1.npz'))
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]


    elif model_name == 'bit_resnext101_1':
        bit_variant = 'BiT-M-R101x1'
        model = bit_models.KNOWN_MODELS[bit_variant](head_size=n_classes, zero_head=True)
        if not os.path.isfile('models/BiT-M-R101x1.npz'):
            print('downloading bit_resnext101_1 weights:')
            os.system('wget https://storage.googleapis.com/bit_models/BiT-M-R101x1.npz -P models/')
        model.load_from(np.load('models/BiT-M-R101x1.npz'))
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]


    elif model_name == 'bit_resnext50_1_DR':
        bit_variant = 'BiT-M-R50x1'
        # in order to use full pretraining we would need n_classes=5 and zero_head=False
        model = bit_models.KNOWN_MODELS[bit_variant](head_size=n_classes, zero_head=True)
        if not os.path.isfile('models/BiT-M-R50x1-run0-diabetic_retinopathy.npz'):
            print('downloading bit_resnext50_1 weights:')
            os.system('wget https://storage.googleapis.com/bit_models/vtab/BiT-M-R50x1-run0-diabetic_retinopathy.npz -P models/')
        model.load_from(np.load('models/BiT-M-R50x1-run0-diabetic_retinopathy.npz'))
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    elif model_name == 'bit_resnext50_1_DR_MOD':
        bit_variant = 'BiT-M-R50x1'
        # in order to use full pretraining we would need n_classes=5 and zero_head=False
        model = bit_models_MOD.KNOWN_MODELS[bit_variant](head_size=n_classes, zero_head=True)
        if not os.path.isfile('models/BiT-M-R50x1-run0-diabetic_retinopathy.npz'):
            print('downloading bit_resnext50_1 weights:')
            os.system('wget https://storage.googleapis.com/bit_models/vtab/BiT-M-R50x1-run0-diabetic_retinopathy.npz -P models/')
        model.load_from(np.load('models/BiT-M-R50x1-run0-diabetic_retinopathy.npz'))
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    else:
        sys.exit('not a valid model_name, check models.get_model.py')
    setattr(model, 'n_classes', n_classes)
    setattr(model, 'cum_derivative_left', cum_derivative_left)

    return model, mean, std


