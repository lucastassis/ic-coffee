'''
Edited by Lucas Tassis

This code is originally from: https://github.com/esgario/lara2018/ by 
@author: Guilherme Esgario
@email: guilherme.esgario@gmail.com

However i made some changes to run multiple instances, if you want the original, check out the link!
'''

import cv2
import numpy as np
import imageio as m
from PIL import Image
from skimage import measure
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
from net_models import PSPNet, resnet50
import sys
np.set_printoptions(threshold=sys.maxsize)

# BIOTIC_STRESS_LABEL = ('Outros', 'Bicho mineiro', 'Ferrugem', 'Mancha-de-phoma', 'Cercosporiose')
BIOTIC_STRESS_LABEL = ('Others', 'Leaf miner', 'Rust', 'Brown Leaf Spot', 'Cercospora Leaf Spot')

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    return resized

class SymptomsImages():
    
    def __init__(self, labels_pred, orig_img):
        self.orig_img = orig_img
        self.labels, self.num_labels = measure.label(labels_pred>1, return_num=True)
        self.leaf_area = np.sum(labels_pred>0)
        self.props = measure.regionprops(self.labels)
        self.i = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.i >= self.num_labels:
            raise StopIteration
        else:
            self.i += 1
            
            # Avoid small symptoms            
            while (self.props[self.i-1].area / self.leaf_area) * 100 < 0.1:
                if self.i >= self.num_labels:
                    raise StopIteration
                    
                self.i += 1
            
            bbox = np.array(self.props[self.i-1].bbox)
            bbox_copy = bbox.copy()
            
            rrow = self.orig_img.shape[0] / self.labels.shape[0]
            rcol = self.orig_img.shape[1] / self.labels.shape[1]
            
            # Padding
            srow = int((bbox[2] - bbox[0]) * 0.2)
            scol = int((bbox[3] - bbox[1]) * 0.2)
            
            bbox[0], bbox[1] = max(0, bbox[0]*rrow - srow), max(0, bbox[1]*rcol - scol)
            bbox[2], bbox[3] = min(self.orig_img.shape[0], bbox[2]*rrow + srow), min(self.orig_img.shape[1], bbox[3]*rcol + scol)
            
            image = self.orig_img[bbox[0]:bbox[2],bbox[1]:bbox[3],:]
            severity = (self.props[self.i-1].area / self.leaf_area) * 100
            
            return image, severity, self.labels, self.i, bbox_copy

class Classifier():
    
    def __init__(self, options):
        self.opt = options
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.results = []
        self.device = 'cpu'

        self.output_img = m.imread(self.opt.background_image)
        self.output_img = self.output_img[:,:,:3]
        self.output_img = np.array(self.output_img, dtype=np.uint8)
        self.width, self.height = 512, 1024
        #self.output_img = image_resize(self.output_img, width=self.width)

    
    def readImage(self, img_size=(1024, 512), show_img=False):
        # Read Image
        img = m.imread(self.opt.in_image_path)
        img = img[:,:,:3]
        img = np.array(img, dtype=np.uint8)
        img = cv2.resize(img, (img_size[0], img_size[1]))

        orig_img = img.copy()

        if show_img:
            plt.imshow(img)
            plt.show()
        
        # Transform
        img = img[:, :, ::-1] # RGB -> BGR
        img = img.astype(np.float64)
        img = img.astype(float) / 255.0
        img = img.transpose(2, 0, 1) # NHWC -> NCHW
        img = torch.from_numpy(img[None,:,:,:]).float()

        # Read background img (Change by Lucas Tassis)
        bg_img = m.imread(self.opt.background_image)
        bg_img = bg_img[:,:,:3]
        bg_img = np.array(bg_img, dtype=np.uint8)
                
        return img.to(self.device), orig_img, bg_img
    

    def showMask(self, mask):
        n_classes = 3
        label_colours = [[0, 0, 0],
                        [0, 176, 0],
                        [255, 0, 0]]
    
        r = mask.copy()
        g = mask.copy()
        b = mask.copy()
        for l in range(0, n_classes):
            r[mask == l] = label_colours[l][0]
            g[mask == l] = label_colours[l][1]
            b[mask == l] = label_colours[l][2]
    
        rgb = np.zeros((mask.shape[0], mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        
        plt.imshow(rgb)
        plt.show()
        
        return rgb
    
    
    def createOutputImage(self, labels, all_leafs):

        # labels = labels.replace('Bicho mineiro', 'Leaf miner')

        # Parameters
        alpha = 0.4        
        kernel = np.ones((5, 5),np.uint8)
        img_width = 800
        colors = [[0, 0, 0], # Black
                  [255, 0, 0], # Red
                  [255, 255, 0], # Yellow
                  [0, 255, 255], # Cyan
                  [127, 0, 255]] # Purple        
        
        # Resize image
        image_orig = self.bg_img.copy()
        # print(f'shape img original = {image_orig.shape}')
        # image_orig = image_resize(image_orig, width=self.width)
        # image = self.bg_img.copy()
        # image = image_resize(image, width=img_width)
        # print(f'shape img original = {image_orig.shape}')
        # print(f'shape output_img = {self.output_img.shape}')
        
        # Resize label
        labels = cv2.resize(labels, (self.output_img.shape[1], self.output_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        labels_contour = cv2.dilate(labels, kernel, iterations=1)
        labels_contour -= labels

        # Resize leafs
        all_leafs = cv2.resize(all_leafs, (self.output_img.shape[1], self.output_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        image_orig[np.where(all_leafs != 1)] = 0

        for i, color in enumerate(colors):
            self.output_img[np.where(labels==i+1)] = color
        
        self.output_img = self.output_img.astype(np.float32)*alpha + image_orig.astype(np.float32)*(1-alpha)
        
        for i, color in enumerate(colors):
            self.output_img[np.where(labels_contour==i+1)] = color
            
        self.output_img = self.output_img.astype(np.uint8)
    

    def segmentation(self):
        # Loading network weights
        model = PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024)
        model.load_state_dict(torch.load(self.opt.segmentation_weights, map_location='cuda'))
        model.to(self.device)
        model.eval()
        out, out_cls = model(self.img.to(self.device))
        
        labels_pred = torch.max(out, 1)[1]
        #self.showMask(labels_pred.cpu().numpy()[0])
        
        return labels_pred.cpu().numpy()[0]
    
    def save(self, image, severity, bbox):
        
        row_p = image.shape[0] / self.width
        col_p = image.shape[1] / self.height
        
        # Save results
        colors = ['black', 'red', 'gold', 'cyan', 'purple']
        fig = plt.figure(figsize=(6, 6*image.shape[0]/image.shape[1]))
        plt.imshow(image)
        plt.axis('off')
        f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]
        
        handles = []
        labels = []
        for i, s in enumerate(severity):
            if s > 0:
                handles.append(f("s", colors[i]))
                labels.append('%s: %.1f%%' % (BIOTIC_STRESS_LABEL[i], s))
        
        # Plot Numbers
        for i, b in enumerate(bbox):
            t = plt.text(0.5*(b[3]+b[1])*col_p-8, b[0]*row_p-10,
                     str(i))
            t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='white', pad=1))
        
        # Plot Legend
        legend = plt.legend(handles,
                            labels,
                            ncol=2,
                            labelspacing=0.5,
                            borderaxespad=0,
                            columnspacing=0.2,
                            handletextpad=0.1,
                            bbox_to_anchor=(1., -0.03),
                            fontsize=12)
        plt.show()
        fig.savefig(self.opt.out_image_path, bbox_inches='tight', pad_inches=0.02, dpi=200)
        plt.close(fig)
    
    
    def run_multiple_imgs(self):

        all_masks = np.zeros((self.width, self.height))
        all_severity = np.zeros(5)
        all_bbox = []
        all_leafs = np.zeros((self.width, self.height))

        for img_path in self.opt.instances_path_list:
            print(img_path)
            self.device = torch.device("cuda")
            
            self.opt.in_image_path = img_path
            
            # Reading image
            self.img, self.orig_img, self.bg_img = self.readImage()
            
            # Segmentation
            labels = self.segmentation()
            all_leafs += labels            
            
            # Classification
            model = resnet50(num_classes=5)
            model.load_state_dict(torch.load(self.opt.symptom_weights, map_location=self.device))
            model.eval()
            model.to(self.device)
            
            # Vector with the total severity of each symptom
            severity = np.zeros(5)
            bbox = []
            new_labels = np.zeros(labels.shape)
            
            result = ''
            for i, (image, severity_ind, lbl, lbl_id, bbox_ind) in enumerate(SymptomsImages(labels, self.orig_img)):

                image = Image.fromarray(image) # Convert to PIL image
                image = self.transform(image.convert('RGB')).view(1, 3, 224, 224)
                image = image.to(self.device)
                out = model(image)
                
                out = torch.nn.functional.softmax(out, dim=1)
                confidence_level, idx = torch.max(out, 1)
                confidence_level = (confidence_level.detach().cpu().numpy() * 100).astype('float')
                
                # Output label
                out = int(idx.cpu().numpy())
                
                severity[out] += severity_ind
                
                new_labels[np.where(np.array(lbl)==lbl_id)] = out+1
                
                # Saving bounding box values
                bbox.append(bbox_ind)
            
            all_masks += new_labels
            all_severity += severity
            all_bbox += bbox
            all_leafs[all_leafs > 1] = 1

        self.createOutputImage(all_masks, all_leafs)
        self.save(self.output_img, all_severity, all_bbox)
       
        return result