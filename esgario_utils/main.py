'''
Edited by Lucas Tassis

This code is originally from: https://github.com/esgario/lara2018/ by 
@author: Guilherme Esgario
@email: guilherme.esgario@gmail.com

However i made some changes to run multiple instances, if you want the original, check out the link!
'''

import os
from classifier import Classifier
import glob

import warnings
warnings.filterwarnings("ignore")

def run(background_image_path, out_image_path, instances_folder_path):
    PATH = os.path.dirname(__file__)
    
    # Parameters
    class Opt():
        pass
    
    options = Opt()
    options.in_image_path = background_image_path
    options.out_image_path = out_image_path
    options.segmentation_weights = os.path.join(PATH, 'net_weights/segmentation.pth')
    options.symptom_weights = os.path.join(PATH, 'net_weights/symptom.pth')

    # Change by Lucas Tassis (lucaswfitassis@gmail.com)
    instances_path_list = glob.glob(instances_folder_path + '/*.jpg')
    options.background_image = background_image_path
    options.instances_path_list = instances_path_list
    
    # Classifier
    clf = Classifier(options)
    result = clf.run_multiple_imgs()

    return result