# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 15:09:23 2020

This script integrates the core functions of Mask_RCNN (https://github.com/matterport/Mask_RCNN) 
into a single file, with modifications made on dataset handling for VoTT based annotations, 
data agumentation and model training, for weed segmentation.

@author: YL
"""

import os
import json
import scipy
import skimage
import numpy as np
import imgaug
import matplotlib.pyplot as plt
from matplotlib import patches
from skimage.measure import find_contours
from skimage.draw import polygon

###############################################################################
# Dataset
###############################################################################
class Dataset():
    """
    The base class for create dataset classes
    """
    def __init__(self):
        self._image_ids = []
        self.image_info = [] #to be updated by add_image
        #set background as the first class
        self.class_info = [{"source": "","id":0, "name": "BG"}] #to be updated by add_class
        self.source_class_ids = {}
    
    def add_image(self, source, image_id, path, **kwargs):
        """Add image and annotations to the dataset"""
        image_info = {"id":image_id, "source":source, "path":path}
        image_info.update(kwargs)
        self.image_info.append(image_info)
        
    def add_class(self, source, class_id, class_name):
        """add a new class into the dataset """
        assert "." not in source, "source name cannot contain a dot"
        #check if the class has been already added
        for info in self.class_info:
            if info["source"]==source and info["id"]==class_id:
                return
        self.class_info.append({"source":source, "id":class_id, "name": class_name})
      
    def load_image(self,image_id):
        """load the specified image and return [H,W,3] numpy array"""
        image = skimage.io.imread(self.image_info[image_id]["path"])
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        if image.shape[-1] == 4:
            image = image[:,:,:3]
        return image
    
    # def load_mask(self, image_id):
    #     """
    #     load instance masks for the given image and return them in the form of an array of binary masks
    #     Return:
    #         masks: a bool array of shape [H, W, instance count] with a binary mask per instance
    #         class_ids: a 1D array of class IDs of the instance masks
        
    #     see the definition inside the subclass WeedDataset defined below
    #     """
    
    def prepare(self):
        """prepare dataset for use"""
        # def clean_names(names):
        #     return ",".join(name.split(",")[:1])
        
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [c["name"] for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)
        
        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)
    
    @property
    def image_ids(self):
        return self._image_ids
        
 
class WeedDataset(Dataset):
    def load_VIA_annotation(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        image_source = "weed"
        self.add_class(image_source, 1, "balloon")
    
        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
    
        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys
    
        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]
    
        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 
    
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            import skimage.io
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
    
            self.add_image(
                image_source,
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                class_ids = np.ones(len(polygons)), # new changes relative to the orginal code in MRCNN/SAMPLE/BALLON
                polygons=polygons)
    
    def renameVoTTJsonAsset(self, srcAsset):
        """
        Rename the name of anotation file exported by VoTT, so as to match up with the name of the associated image
        VoTT names the asset file using the internally generated IDs that are not descriptive to users
        srcAsset: the absolute path to the annotation file to directory where the files are stored  """   
        if not os.path.isdir(srcAsset):
            print("\n Rename the file:{}".format(srcAsset))
            with open(srcAsset,'r') as f:
                A = json.load(f)
            imgName = A["asset"]["name"]
            imgName = imgName.replace("%20", ' ')
            imgName,_ = os.path.splitext(imgName)
            dstName = os.path.join(os.path.dirname(srcAsset),imgName+'.json')
            print(dstName)
            os.rename(srcAsset,dstName)
        else:
            jsonFiles = os.listdir(srcAsset)
            for jsonFile in jsonFiles:
                if jsonFile.endswith(".json"):
                    #print(f"fileName:{jsonFile}")
                    with open(os.path.join(srcAsset,jsonFile),'r') as f:
                        A = json.load(f)
                    imgName = A["asset"]["name"]
                    imgName = imgName.replace("%20", ' ')
                    imgName,_ = os.path.splitext(imgName)
                    srcName = os.path.join(srcAsset,jsonFile)
                    dstName = os.path.join(srcAsset,imgName+'.json')
                    os.rename(srcName,dstName)
              
    def parse_vottAsset(self, asset, tags, data_dir=None):
        # extract annotation from a single asset (image) generated by VoTT
        imgName = asset["asset"]["name"]
        imgSize = list(asset["asset"]["size"].values())
        
        #replace '%20' by ' ' (if any) due to the presence of whitespaces
        imgName = imgName.replace("%20"," ")
        
        if data_dir != None:
            imgPath = os.path.join(data_dir,imgName)
        else:
            imgPath = asset["asset"]["path"][5:]
        imgPath = imgPath.replace("%20"," ")    
        
        imgRegions = asset["regions"] 
        class_names = []
        polygons = []
        boundingBoxes = []
        for roi in imgRegions:
            objTag = roi["tags"][0]
            objBoundingBox = roi["boundingBox"] #{"height': ,'width': ,'left': ,'top':}
            objPoints = roi["points"] #N-element dict, each corresponding to a dict {'x': ,'y': }
            objXPoints = list(point['x'] for point in objPoints)
            objYPoints = list(point['y'] for point in objPoints)
            objPolygon = {'all_points_x':objXPoints, 'all_points_y':objYPoints}
            
            class_names.append(objTag)
            polygons.append(objPolygon)
            boundingBoxes.append(objBoundingBox)
                
        # class_ids
        class_ids = [tags.index(class_name) for class_name in class_names]
        
        # Summary
        annot_info = {"image_id": imgName, 
                      "image_path":imgPath, 
                      "width":imgSize[0],"height":imgSize[1],
                      "polygons":polygons,
                      "boxes": boundingBoxes,
                      "class_names":class_names,
                      "class_ids": class_ids}
        return annot_info
    
    def load_VoTT_annotation(self, annot_dir, annot_file, tags, data_dir=None,
                              subset=None, return_annot=False):
        """load VoTT annotations (.json) and add them to the dataset"""
        # Subset provovided
        if subset is not None:
            assert subset in ["train", "val", "test"]
            annot_dir = os.path.join(annot_dir, subset)
            
        # json file checking
        if not annot_file:
            annotations = []
            for file in os.listdir(annot_dir):
                if file.endswith(".json"):
                    annotations.append(json.load(open(os.path.join(annot_dir, file))))
        else:
            if type(annot_file) == str:
                annotations = json.load(open(os.path.join(annot_dir, annot_file)))
            elif type(annot_file) == list:
                annotations = []
                for temp_file in annot_file:
                    annotations.append(json.load(open(os.path.join(annot_dir, temp_file))))                    
        
        annot_INFO = []
        if len(annotations) == 9 and (type(annotations) is dict):
            assets = list(annotations["assets"].values())            
            for a in assets:
                annot_INFO.append(self.parse_vottAsset(a,tags,data_dir))  
        elif len(annotations) == 3 and (type(annotations) is dict):
            annot_INFO.append(self.parse_vottAsset(annotations,tags,data_dir))
        elif type(annotations) is list:
            for annotation in annotations:
                annot_INFO.append(self.parse_vottAsset(annotation,tags,data_dir))    
        else:
            print("Unknonw type of asset!\n")
            pass
        
        # add images 
        for annot in annot_INFO:
            self.add_image("weed",
                    image_id = annot["image_id"],
                    path = annot["image_path"],
                    width = annot["width"],
                    height = annot["height"],
                    polygons = annot["polygons"],
                    class_ids = annot["class_ids"],
                    class_names = annot["class_names"])
            for ii in range(len(annot["class_ids"])):
                self.add_class("weed",annot["class_ids"][ii],annot["class_names"][ii])
        
        # add class
        # for ii in range(len(tags)):
        #     self.add_class("weed",ii,tags[ii])
        
        if return_annot:
            return annot_INFO 
        
    def load_mask(self, image_id):
        return self.load_VoTT_mask(image_id)
    
    def load_VoTT_mask(self,image_id):
        """
        load instance mask given an image
        Return a boolean mask array and corresponding class ID
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "weed":
            print("\n Unexpected image source!")
            return
        
        # convert polygon to a binary mask
        mask = np.zeros([image_info["height"], image_info["width"], len(image_info["polygons"])], 
                        dtype=np.uint8)
        for i, p in enumerate(image_info["polygons"]):
            rr, cc = polygon(p['all_points_y'], p['all_points_x']) # pixels inside the polygon
            mask[rr,cc,i] = 1
        
        # class ids
        class_ids = image_info["class_ids"]
        
        return mask.astype(np.bool), np.array(class_ids, dtype=np.int32)
    
    def get_image_path(self,image_id):
        """Return the path of the image"""
        info = self.image_info[image_id]
        if info["source"] == "weed":
            return info["path"]

###############################################################################
# Utilities
###############################################################################
def extract_boxes(annotINFO, image_id):
    BOX = annotINFO[image_id]["boxes"]
    boxes = []
    for box in BOX:
        xmin = box["left"]
        ymin = box["top"]
        xmax = xmin + box["width"]
        ymax = ymin + box["height"]
        boxes.append([xmin,ymin,xmax,ymax])
    return boxes
 
def comput_boundingboxes(mask):
    """
    Compute bounding boxes based on binary mask
    Return: bbox array [num_instances,(y1,x1,y2,x)]
    """
    boxes = np.zeros([mask.shape[-1],4])
    for i in range(mask.shape[-1]):
        m = mask[:,:,i]
        xIndices = np.where(np.any(m,axis=0))[0]
        yIndices = np.where(np.any(m,axis=1))[0]
        if xIndices[0]:
            xmin, xmax = xIndices[[0,-1]]
            ymin, ymax = yIndices[[0,-1]]
        else:
            # no masks
            xmin, xmax, ymin, ymax = 0, 0, 0, 0
        boxes[i,:] = np.array([ymin,xmin,ymax,xmax])
    return boxes.astype(np.int32)
 
    
def random_colors(N, bright=True):
    # generate random visually distinct colors
    import colorsys
    import random
    brightness = 1.0 if bright else 0.7
    hsv = [(i/N,1,brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors
 
def apply_mask(image,mask,color,alpha=0.5):
    """apply the mask to the image for visual display"""
    for c in range(3):
        image[:,:,c] = np.where(mask==1, image[:,:,c]*(1-alpha)+alpha*color[c]*255, image[:,:,c])
    return image   
    
    
def visualize_instances(image, image_mask, boxes=None, class_names=None, figsize=(16,16), colors=None):
    """
    Visuallize semantic instances annotated by VIA in the image
    image: rbg image
    imag_mask: [height, width, object#]
    fisize: optional, for image display
    colors: optional, color settings for object distinction
    """
    import matplotlib.pyplot as plt
    from matplotlib import patches
    _, ax = plt.subplots(1, figsize=figsize)
    ax.axis('off')
    
    N  = image_mask.shape[-1] # number of objects
    
    #color settings
    if colors is None:
        import random 
        import colorsys
        hsv = [(i/N, 1, 1) for i in range(N)]
        colors = list(map(lambda c:colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
    
    #masked_image = np.copy(image)
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        
        # bounding boxes
        if boxes:
            xmin, ymin, xmax, ymax = boxes[i]
            p = patches.Rectangle((xmin,ymin),xmax-xmin,ymax-ymin, 
                                  linewidth=2, alpha=0.7, linestyle='dashed',
                                  edgecolor=color,facecolor='none')
            ax.add_patch(p)
            
            # add tags
            if class_names:
                label = class_names[i]
                ax.text(xmin,ymin+15,label,color='k',size=15,backgroundcolor='none')            
        
        # masks
        mask = image_mask[:,:,i]        
        for j in range(3):
            masked_image[:,:,j] = np.where(mask==1, masked_image[:,:,j]*0.5+0.5*color[j]*255, 
                                           masked_image[:,:,j])
        
        #draw polygons
        contours = find_contours(mask,0.5)
        for verts in contours:
            verts = np.fliplr(verts)
            p = patches.Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
           
    ax.imshow(masked_image.astype(np.uint8))
    plt.show()

def draw_boxes(image, boxes, masks=None, captions=None, 
               visibilities=None, title="", ax=None):
    """Draw bounding box and or segmentatiion masks with customizations
    boxes = [instance_count (y1, x1, y2, x2, class_id)]
    masks = [height, width, N]
    """
    import matplotlib.pyplot as plt
    from matplotlib import patches
    N = boxes.shape[0]
    if not ax:
        _, ax = plt.subplots(1, figsize=(12,12))
    
    #generate random colors
    colors = random_colors(N)
    
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        visibility = visibilities[i] if visibilities is not None else 1
        if visibility == 0:
            color = 'gray'
            style = 'dotted'
            alpha = 0.5
        elif visibility == 1:
            color = colors[i]
            style = 'dotted'
            alpha = 1
        elif visibility == 2:
            color = colors[i]
            style = 'solid'
            alpha = 1
    
        # boxes
        if not np.any(boxes[i]):
            continue # no boxes
        y1, x1, y2, x2 = boxes[i]
        p = patches.Rectangle((x1,y1), x2-x1, y2-y1, linewidth=2, 
                              alpha=alpha, linestyle=style,
                              edgecolor=color, facecolor='none')
        ax.add_patch(p)
        
        # captions
        if captions is not None:
            caption = captions[i]
            ax.text(x1,y1,caption,size=11, verticalalignment='top',color='w',backgroundcolor="none",
                    bbox={'facecolor':color,'alpha':0.5,'pad':2,'edgecolor':'none'})
        
        # masks
        from skimage.measure import find_contours
        if masks is not None:
            mask = masks[:,:,i]
            masked_image = apply_mask(masked_image, mask, color)
            # mask polygon
            padded_mask = np.zeros((mask.shape[0]+2, mask.shape[1]+2), dtype=np.uint8)
            padded_mask[1:-1,1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # subtract padding and flip (y,x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = patches.Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
        ax.imshow(masked_image.astype(np.uint8))
                

def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode='square'):    
    """Resizes an image keeping the aspect ratio unchanged.
     min_dim: if provided, resizes the image such that it's smaller dimension == min_dim
     max_dim: if provided, ensures that the image longest side doesn't exceed this value
     
     min_scale: if provided, ensure that the image is scaled up by at least
         this percent even if min_dim doesn't require it.
         
     mode: Resizing mode.
         none: No resizing. Return the image unchanged.
         square: Resize and pad with zeros to get a square image of size [max_dim, max_dim].
         pad64: Pads width and height with zeros to make them multiples of 64.
                If min_dim or min_scale are provided, it scales the image up
                before padding. max_dim is ignored in this mode.
                The multiple of 64 is needed to ensure smooth scaling of feature
                maps up and down the 6 levels of the FPN pyramid (2**6=64).
         crop: Picks random crops from the image. First, scales the image based
               on min_dim and min_scale, then picks a random crop of
               size min_dim x min_dim. Can be used in training only.
               max_dim is not used in this mode.
    
     Returns:
     image: the resized image
     window: (y1, x1, y2, x2). If max_dim is provided, padding might be inserted in the returned image. If so, this window is the
         coordinates of the image part of the full image (excluding the padding). The x2, y2 pixels are not included.
     scale: The scale factor used to resize the image
     padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
     """
    import skimage
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # Scale?
    if min_dim:
        scale = max(1, min_dim / min(h, w)) # Scale up but not down
    if min_scale and scale < min_scale:
        scale = min_scale
    
     # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max
    
     # Resize image using bilinear interpolation
    if scale != 1:
        image = skimage.transform.resize(image, (round(h * scale), round(w * scale)), 
                                          preserve_range=True)
    
     # Need padding or cropping?
    if mode == "square":
         # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype), window, scale, padding, crop
   
 
def resize_mask(mask, scale, padding, crop=None):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = scipy.ndimage.zoom(mask,zoom=[scale,scale,1], order=0)
    if crop is not None:
        y, x, h, w = crop
        mask = mask[y:y+h, x:x+w]
    else:
        mask = np.pad(mask,padding, mode='constant', constant_values=0)
    return mask

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in all visualizations in the notebook. 
    Provide a central point to control graph sizes.
    Adjust the size attribute to control how big to render images
    """
    import matplotlib.pyplot as plt
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

# random split data into train and validation sets
def train_valid_split(dataDir, validation_split=0.25):
    """split a dataset object into train and valiation datasets
    dataDir = "F:/OpenCV4/DeepLearning/RCNNs/MaskRCNN/weedDataDemo/Carpetweed"
    """
    import os
    import shutil
    import numpy as np
    from itertools import compress
    files = os.listdir(dataDir)
    
    jsonFiles = list(compress(files,[f.endswith(".json") for f in files])) 
    imgFiles = list(compress(files,[f.lower().endswith((".jpeg",".jpg",".png",".tiff")) for f in files]))
                      
    N = len(jsonFiles)
    idx = np.arange(N)
    np.random.seed(0) 
    val_idx = np.random.choice(idx, np.round(N*validation_split).astype(np.int32), replace=False)
    train_idx = np.setdiff1d(idx,val_idx)
    jsonFiles_train = [jsonFiles[i] for i in train_idx] 
    imgFiles_train = [imgFiles[i] for i in train_idx]
    jsonFiles_val = [jsonFiles[i] for i in val_idx]
    imgFiles_val = [imgFiles[i] for i in val_idx]
    return imgFiles_train, jsonFiles_train, imgFiles_val, jsonFiles_val


###############################################################################
# Data generator
###############################################################################
def load_image_gt_da(dataset, config, image_id, augmentation=False, use_mini_mask=False):
    """
    Load ground truth image data with data augmentation
    See also mrcnn.model.load_image_gt

    Parameters
    ----------
    dataset : Dataset instance
    config : Config instance
    image_id : image ID
    augmentation : optional, an imgaug augmenter (e.g., imgaug.augmenters.Affine(rotate=(-45, 45)))
    use_mini_mask : if false, return full_size masks and images; otherwise resized
                    to MINI_MASK_SHAPE

    Returns
    -------
    image: [height, width, 3]
    mask: [height, width, instance_count] #height and width are those of the image unless use_mini_mask is True
    """

    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    
    original_shape = image.shape
    image, window, scale, padding, crop = resize_image(image,
                                                       min_dim=config.IMAGE_MIN_DIM,
                                                       min_scale=config.IMAGE_MIN_SCALE,
                                                       max_dim=config.IMAGE_MAX_DIM,
                                                       mode=config.IMAGE_RESIZE_MODE)
    mask = resize_mask(mask, scale, padding, crop)
    
    if augmentation:
        # activator (None or callable, optional) – 
        # A function that gives permission to execute an augmenter.
        # ``f(images, augmenter, parents, default)``
        def hook(images, augmenter, parents, default):    
            MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                                   "Fliplr", "Flipud", "CropAndPad",
                                   "Affine", "PiecewiseAffine"]
            return augmenter.__class__.__name__ in MASK_AUGMENTERS
        
        # before augmentation
        image_shape = image.shape
        mask_shape = mask.shape
        
        # Convert this augmenter from a stochastic to a deterministic one.
        # Thus, the same augmentation is applied to the image and mask
        det = augmentation.to_deterministic() # imgaug.augmenters.arithmetic
        image = det.augment_image(image)
 
        # https://imgaug.readthedocs.io/en/latest/source/api_augmenters_meta.html#imgaug.augmenters.meta.Augmenter.augment_image
        mask = det.augment_image(mask.astype(np.uint8), 
                                 hooks=imgaug.HooksImages(activator=hook))
        
        # verify if image size does not change
        assert image.shape==image_shape, "Augmentation shouldn't change image size"
        assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
        # change back mask type
        mask = mask.astype(np.bool)
        print("\n Augmentation is Done!\n")
        
    # filter out the box when the object mask is fully cropped out
    _idx = np.sum(mask,axis=(0,1))>0
    mask = mask[:,:,_idx]
    class_ids = class_ids[_idx]
    bboxes = comput_boundingboxes(mask)
    
    # track the classes supported in the dataset
    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1
    
    # Resize masks to smaller size to reduce memory usage
    if use_mini_mask:
        mask = minimize_mask(bboxes, mask, config.MINI_MASK_SHAPE)
    
    # Image meta data
    from mrcnn.model import compose_image_meta
    image_meta = compose_image_meta(image_id, original_shape, image.shape,
                                    window, scale, active_class_ids)
    return image, image_meta, class_ids.astype(np.int32), bboxes, mask
 
## minimize_mask()
def minimize_mask(bbox, mask, mini_shape):
    """Resize masks to a smaller version to reduce memory load. Mini-masks can be resized back to image scale using expand_masks()

    See inspect_data.ipynb notebook for more details.
    """
    import skimage
    mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        # Pick slice and cast to bool in case load_mask() returned wrong dtype
        m = mask[:, :, i].astype(bool)
        y1, x1, y2, x2 = bbox[i][:4]
        m = m[y1:y2, x1:x2]
        if m.size == 0:
            raise Exception("Invalid bounding box with area of zero")
        # Resize with bilinear interpolation
        m = skimage.transform.resize(m, mini_shape, order=1, mode='constant', cval=0, clip=True)
        mini_mask[:, :, i] = np.around(m).astype(np.bool)
    return mini_mask

def mold_image(images, config):
    # meaning centering
    return images.astype(np.float32) - config.MEAN_PIXEL

def unmold_image(normalized_images, config):
    return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)


def data_generator(dataset, config, shuffle=True, augmentation=None, random_rois=0,
                   batch_size=1, detection_targets=False, no_augmentation_sources=None):
    """
    A generator that returns images and corresponding target class ids, bounding box deltas and masks

    Parameters
    ----------
    dataset : TYPE
    config : TYPE
    shuffle : TYPE, optional
    augmentation : TYPE, optional
    random_rois : TYPE, optional
    batch_size : TYPE, optional
    detection_targets : TYPE, optional
    no_augmentation_sources : list of sources to exclude for augmentation (a string list)
    """
    b = 0
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0
    no_augmentation_sources = no_augmentation_sources or []
    
    #Anchors
    from mrcnn.model import compute_backbone_shapes
    from mrcnn.utils import generate_pyramid_anchors
    backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)
    anchors = generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                       config.RPN_ANCHOR_RATIOS,
                                       backbone_shapes,
                                       config.BACKBONE_STRIDES,
                                       config.RPN_ANCHOR_STRIDE)
    
    # keras generator
    from mrcnn.model import build_rpn_targets
    while True:
        try:
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)
            
            # use load_image_gt_da instead of the original load_image_gt
            image_id = image_ids[image_index]
            if dataset.image_info[image_id]['source'] in no_augmentation_sources:
                image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                    load_image_gt_da(dataset, config, image_id, augmentation=None,
                                     use_mini_mask=config.USE_MINI_MASK)
            else:
                image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                    load_image_gt_da(dataset, config, image_id, augmentation=augmentation,
                                     use_mini_mask=config.USE_MINI_MASK)
            
            if not np.any(gt_class_ids>0):
                continue #background class
            
            # RPN targets
            rpn_match, rpn_bbox = build_rpn_targets(image.shape, anchors,
                                                    gt_class_ids, gt_boxes, config)
            # mrcnn targets
            from mrcnn.model import generate_random_rois, build_detection_targets
            if random_rois:
                rpn_rois = generate_random_rois(image.shape, random_rois, gt_class_ids, gt_boxes)
                if detection_targets:
                    rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask =\
                        build_detection_targets(rpn_rois, gt_class_ids, gt_boxes, gt_masks, config)
            
            # Init batch arrays
            if b == 0:
                batch_image_meta = np.zeros((batch_size,) + image_meta.shape, 
                                            dtype=image_meta.dtype)
                batch_rpn_match = np.zeros([batch_size, anchors.shape[0], 1], 
                                           dtype=rpn_match.dtype)
                batch_rpn_bbox = np.zeros([batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], 
                                          dtype=rpn_bbox.dtype)
                
                batch_images = np.zeros((batch_size,) + image.shape, dtype=np.float32)
                batch_gt_class_ids = np.zeros((batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
                batch_gt_boxes = np.zeros((batch_size, config.MAX_GT_INSTANCES, 4), dtype=np.int32)
                batch_gt_masks = np.zeros((batch_size, gt_masks.shape[0], gt_masks.shape[1],
                                           config.MAX_GT_INSTANCES), dtype=gt_masks.dtype)
                if random_rois:
                    batch_rpn_rois = np.zeros((batch_size, rpn_rois.shape[0], 4), dtype=rpn_rois.dtype)
                    if detection_targets:
                        batch_rois = np.zeros((batch_size,) + rois.shape, dtype=rois.dtype)
                        batch_mrcnn_class_ids = np.zeros((batch_size,) + mrcnn_class_ids.shape, 
                                                         dtype=mrcnn_class_ids.dtype)
                        batch_mrcnn_bbox = np.zeros((batch_size,) + mrcnn_bbox.shape, 
                                                    dtype=mrcnn_bbox.dtype)
                        batch_mrcnn_mask = np.zeros((batch_size,) + mrcnn_mask.shape, 
                                                    dtype=mrcnn_mask.dtype)    
                        
            # If more instances than fits in the array, sub-sample from them.
            if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
                ids = np.random.choice(np.arange(gt_boxes.shape[0]), 
                                       config.MAX_GT_INSTANCES, 
                                       replace=False)
                gt_class_ids = gt_class_ids[ids]
                gt_boxes = gt_boxes[ids]
                gt_masks = gt_masks[:, :, ids]
            
            # add to batch
            batch_image_meta[b] = image_meta
            batch_rpn_match[b] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[b] = rpn_bbox
            batch_images[b] = mold_image(image.astype(np.float32), config) #mold_image
            batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
            batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
            batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks
            if random_rois:
                batch_rpn_rois[b] = rpn_rois
                if detection_targets:
                    batch_rois[b] = rois
                    batch_mrcnn_class_ids[b] = mrcnn_class_ids
                    batch_mrcnn_bbox[b] = mrcnn_bbox
                    batch_mrcnn_mask[b] = mrcnn_mask
            b += 1
            
            # batch full
            if b >= batch_size:
                inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
                          batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
                outputs = []

                if random_rois:
                    inputs.extend([batch_rpn_rois])
                    if detection_targets:
                        inputs.extend([batch_rois])
                        # Keras requires that output and targets have the same number of dimensions
                        batch_mrcnn_class_ids = np.expand_dims(batch_mrcnn_class_ids, -1)
                        outputs.extend([batch_mrcnn_class_ids, batch_mrcnn_bbox, 
                                        batch_mrcnn_mask])

                yield inputs, outputs
                
                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # log it and skip the image
            import logging
            logging.exception("Error processing image {}".format(dataset.image_info[image_id]))
            error_count += 1
            if error_count>5:
                raise
                
###############################################################################
# Mask_RCNN
###############################################################################
import mrcnn.model
from mrcnn.model import MaskRCNN
class Mask_RCNN(MaskRCNN):
    """
    Create a Mask_RCNN model
    """
    def model_train(self, train_dataset, learning_rate, epochs, layers=None,
                    val_dataset=None, 
                    augmentation=None, 
                    custom_callbacks=None, 
                    no_augmentation_sources=None):
        # see also mrcnn.model.train()
        # https://www.geeksforgeeks.org/keras-fit-and-keras-fit_generator/
        # https://github.com/jingweimo/keras/blob/master/keras/models.py#L1028
        layer_regex = {
            # all layers but the backbone
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]
        
        # Data generators
        train_generator = mrcnn.model.data_generator(train_dataset, 
                                                     self.config, 
                                                     shuffle=True,
                                                     augmentation=augmentation,
                                                     batch_size=self.config.BATCH_SIZE,
                                                     no_augmentation_sources=no_augmentation_sources)
        if val_dataset is not None:
            val_generator = mrcnn.model.data_generator(val_dataset, 
                                                       self.config, 
                                                       shuffle=True,
                                                       batch_size=self.config.BATCH_SIZE)
        
        # Create log_dir if it does not exist
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
             
        # Base callbacks (https://keras.io/api/callbacks/)
        callbacks = [keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=0,
                     write_graph=True, write_images=False),
                     keras.callbacks.ModelCheckpoint(self.checkpoint_path, verbose=0, 
                                                     save_weights_only=True)]
        # add custom callbacks (https://keras.io/guides/writing_your_own_callbacks/)
        if custom_callbacks:
            callbacks += custom_callbacks               
        
        # Train
        print("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)
        
        # Work-around for Windows: Keras fails on Windows when using multiprocessing workers. 
        # See discussion here: https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        import multiprocessing
        if os.name is 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()
        
        if val_dataset is not None:
            self.keras_model.fit_generator(train_generator,
                                           initial_epoch=0,
                                           epochs=epochs,
                                           steps_per_epoch=self.config.STEPS_PER_EPOCH,
                                           callbacks=callbacks,
                                           validation_data=val_generator,
                                           validation_steps=self.config.VALIDATION_STEPS,
                                           max_queue_size=100,
                                           workers=workers,
                                           use_multiprocessing=True)   
        else:
           self.keras_model.fit_generator(train_generator,
                                          initial_epoch=0,
                                          epochs=epochs,
                                          steps_per_epoch=self.config.STEPS_PER_EPOCH,
                                          callbacks=callbacks,
                                          max_queue_size=100,
                                          workers=workers,
                                          use_multiprocessing=True)
        self.epoch = max(self.epoch, epochs)
        
