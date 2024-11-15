# --------------------------------------------------------
# Dual Cross Attention
# Copyright (c) 2023 Gorkem Can Ates
# Licensed under The MIT License [see LICENSE for details]
# Written by Gorkem Can Ates (gca45@miami.edu)
# --------------------------------------------------------

import os
import torch
from torch.optim import Adam
from trainer.trainer import MainTrainer
from model.unet import Unet
from model.res_unet_plus import ResUnetPlus
from model.multi_res_unet import MultiResUnet
from model.r2_unet import R2Unet
from model.vnet import Vnet
from model.double_unet import DoubleUnet
from losses.loss import DiceLoss
from metrics.metrics import IoU, DiceScore
from Dataset.Datasets import DATASET
from transforms.transforms import Transforms
from writer.writer import TensorboardWriter
import time

class Parameters:
    def __init__(self, experiment, file):
        self.experiment = experiment
        self.DEVICE = 'cuda'
        self.file = file
        self.load_file = 'unet_f/'
        self.train_data_dir = 'Dataset/' + self.experiment + 'images/'
        self.train_mask_dir = 'Dataset/' + self.experiment + 'masks/'
        self.LOGDIR = f'runs/' + self.experiment + self.file
        self.FIG_PATH = 'RESULTS/' + self.experiment + self.file + 'images/'
        self.result_SAVEPATH = 'RESULTS/' + self.experiment + self.file + 'metrics/'
        self.result_HISTORYPATH = 'RESULTS/' + self.experiment + self.file + 'history/'
        self.model_SAVEPATH = 'RESULTS/' + self.experiment + self.file + 'models/'
        self.model_LOADPATH = 'RESULTS/' + self.experiment + self.load_file + 'models/'
        self.METRIC_CONDITION = DiceScore.__name__.lower()
        self.TO_TENSORBOARD = True
        self.VALIDATION = True
        self.PRETRAINED = False
        self.FINAL_VALIDATION = False
        self.SAVE_MODEL = True
        self.DEBUG = False
        self.TRANSFORM = True
        self.SHUFFLE = True
        self.NUM_WORKERS = 0
        self.CUDA_COUNT = torch.cuda.device_count()

class HyperParameters:
    def __init__(self, experiment, attention, model):
        self.NUM_EPOCHS = 200
        self.LEARNING_RATE = 0.0001
        self.BETA1 = 0.9
        self.BETA2 = 0.999
        self.RESIZE_SHAPE = (224, 224)
        self.FILTER_COEFF = 0.5
        self.NUM_BLOCKS = 1
        self.ATTENTION = attention
        if model.__name__ == 'ResUnetPlus':
            self.PATCH_SIZE = 4
            self.CHANNEL_HEAD_DIM = [1, 1, 1]
            self.SPATIAL_HEAD_DIM = [4, 4, 4]
        else:
            self.PATCH_SIZE = 8
            self.CHANNEL_HEAD_DIM = [1, 1, 1, 1]
            self.SPATIAL_HEAD_DIM = [4, 4, 4, 4]            
        self.METRIC_CONDITION = 'max'

        if experiment == 'SYNAPS/':
            self.TRANSFORM_MODE = 'cv2'
            self.IN_CHANNELS = 1
            self.NUM_CLASSES = 9
            self.TRAIN_BATCH_SIZE = 24
            self.TEST_BATCH_SIZE = 31                        
        elif experiment == 'Kvasir/':
            self.TRANSFORM_MODE = 'torch'
            self.IN_CHANNELS = 3
            self.NUM_CLASSES = 1
            self.TRAIN_BATCH_SIZE = 16
            self.TEST_BATCH_SIZE = 25                        
        elif experiment == 'GlaS/':
            self.TRANSFORM_MODE = 'torch'
            self.IN_CHANNELS = 3
            self.NUM_CLASSES = 1 
            self.TRAIN_BATCH_SIZE = 4
            self.TEST_BATCH_SIZE = 20       
        elif experiment == 'MoNuSeg/':
            self.TRANSFORM_MODE = 'torch'
            self.IN_CHANNELS = 3
            self.NUM_CLASSES = 1 
            self.TRAIN_BATCH_SIZE = 4
            self.TEST_BATCH_SIZE = 14                       
        elif experiment == 'CVC/':
            self.TRANSFORM_MODE = 'torch'
            self.IN_CHANNELS = 3
            self.NUM_CLASSES = 1 
            self.TRAIN_BATCH_SIZE = 16
            self.TEST_BATCH_SIZE = 17            
        else:
            raise Exception('Experiment not found.') 


class MAIN:
    def __init__(self, experiment, model, attention, file, ca, sa):

        self.params = Parameters(experiment=experiment, 
                                 file=file)

        self.hyperparams = HyperParameters(experiment=experiment, 
                                           attention=attention, 
                                           model=model)

        self.model = model(
                            attention=self.hyperparams.ATTENTION,
                            n=self.hyperparams.NUM_BLOCKS,
                            in_features=self.hyperparams.IN_CHANNELS if model.__name__ != 'DoubleUnet' else 3, 
                            out_features=self.hyperparams.NUM_CLASSES,
                            k=self.hyperparams.FILTER_COEFF,
                            input_size=self.hyperparams.RESIZE_SHAPE, 
                            patch_size=self.hyperparams.PATCH_SIZE,
                            spatial_att=sa,
                            channel_att=ca,
                            spatial_head_dim=self.hyperparams.SPATIAL_HEAD_DIM,
                            channel_head_dim=self.hyperparams.CHANNEL_HEAD_DIM,
                            device=self.params.DEVICE)  
  
        self.metrics = [
                        IoU(num_classes=self.hyperparams.NUM_CLASSES),
                        DiceScore(num_classes=self.hyperparams.NUM_CLASSES)
                        ]


        self.criterion = DiceLoss(num_classes=self.hyperparams.NUM_CLASSES,
                                 )

        self.optimizer = Adam(params=self.model.parameters(),
                              lr=self.hyperparams.LEARNING_RATE,
                              betas=(self.hyperparams.BETA1,
                                     self.hyperparams.BETA2))

        self.transforms = Transforms(shape=self.hyperparams.RESIZE_SHAPE, 
                                     transform=self.params.TRANSFORM, 
                                     mode=self.hyperparams.TRANSFORM_MODE)
           

        self.dataset = DATASET( experiment=self.params.experiment,
                                im_path=self.params.train_data_dir,
                                mask_path=self.params.train_mask_dir,
                                train_transform=self.transforms.train_transform,
                                val_transform=self.transforms.val_transform,
                                num_classes=self.hyperparams.NUM_CLASSES,
                                debug=self.params.DEBUG
                                    )


        self.writer = TensorboardWriter(exp=self.params.experiment,
                                        PATH=self.params.LOGDIR,
                                        fig_path=self.params.FIG_PATH,
                                        num_data=48)


        self.trainer = MainTrainer(model=self.model,
                                   params=self.params,
                                   hyperparams=self.hyperparams,
                                   metrics=self.metrics,
                                   dataset=self.dataset,
                                   optimizer=self.optimizer,
                                   criterion=self.criterion,
                                   writer=self.writer
                                   if self.params.TO_TENSORBOARD else None
                                   )
        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.makedirs()

        with open('RESULTS/' + self.params.experiment + self.params.file + 'model.txt', 'w') as f:
            f.write(self.model.__str__())
        with open('RESULTS/' + self.params.experiment + self.params.file + 'summary.txt', 'w') as f:
            f.write(f'Total model parameters :' "{:,}".format(self.total_params))
            f.write(f'\nMODEL : {self.model._get_name()} ')
            f.write(f'\nPARAMS: {self.params.__dict__} ')    
            f.write(f'\nHYPERPARAMS: {self.hyperparams.__dict__} ')    
            f.write(f'\nCRITERION : {self.criterion._get_name()} ')
            f.write(f'\nBATCH SIZE : {self.hyperparams.TRAIN_BATCH_SIZE} ')
            f.write(f'\nLEARNING RATE : {self.hyperparams.LEARNING_RATE} ')
            f.write(f'\nDEVICE : {self.params.DEVICE.upper()} ')


    def run(self):
        tic = time.perf_counter()
        self.trainer.fit()
        toc = time.perf_counter()
        print(f"TOTAL TRAINING TIME: {(toc - tic) / 60} minutes")


    def validate(self):
        results = self.trainer.validate()
        return results
    
    def makedirs(self):
        self.makedir('RESULTS/')
        self.makedir(os.path.join('RESULTS/', self.params.experiment))
        self.makedir(os.path.join('RESULTS/' + self.params.experiment, self.params.file))
        self.makedir(self.params.result_HISTORYPATH)
        self.makedir(self.params.model_SAVEPATH)
        self.makedir(self.params.result_SAVEPATH)
        if self.params.TO_TENSORBOARD or self.params.FINAL_VALIDATION:
            self.makedir(self.params.FIG_PATH)


    def makedir(self, path):
        if not os.path.exists(path=path):
            os.mkdir(path=path)


if __name__ == '__main__':
    seed = 666
    deterministic = True
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    experiments = [
                    'GlaS/',
                    'MoNuSeg/',
                    'CVC/', 
                    'Kvasir/',
                    'SYNAPS/',
                    ]

    models = [
                Unet,
                Unet,        
                ResUnetPlus, 
                ResUnetPlus, 
                MultiResUnet,
                MultiResUnet,                   
                R2Unet, 
                R2Unet,
                Vnet, 
                Vnet,
                DoubleUnet,
                DoubleUnet

              ]

    files = [
                'unet/',
                'unet_dca/',
                'resunetplus/',
                'resunetplus_dca/',
                'mresunet/',
                'mresunet_dca/',
                'r2unet/',
                'r2unet_dca/',
                'vnet/',
                'vnet_dca/',
                'dunet/',
                'dunet_dca/',

                

              ]
    attentions = [
                False, 
                True,
                False, 
                True,
                False, 
                True,
                False, 
                True,
                False, 
                True,
                False, 
                True,
            ]

    ca = [
                False, 
                True,
                False, 
                True,
                False, 
                True,
                False, 
                True,
                False, 
                True,
                False, 
                True,            
            ]
    sa = [
                False, 
                True,
                False, 
                True,
                False, 
                True,
                False, 
                True,
                False, 
                True,
                False, 
                True,
              ]

    for exp in experiments:
        for model, file, attention, c, s in zip(models, files, attentions, ca, sa):
            trainer = MAIN(experiment=exp, 
                          model=model, 
                          attention=attention, 
                          file=file, 
                          ca=c, 
                          sa=s
                          )
            trainer.run()
