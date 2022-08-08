import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import config
from tensorboardX import SummaryWriter

label_colors_list = [
    (135, 206, 236), # Sky 
    (166, 227, 27), # Bridge
    (74, 194, 154), # Building
    (163, 46, 18), # Wall
    (30, 73, 204), # Column_Pole
    (251, 116, 77), # TrafficCone
    (132, 17, 209), # electric_pole
    (78, 80, 199), # street light
    (23, 125, 202), # trafic light
    (119, 61, 128), # pavedRoad
    (217, 112, 230), # unpaved_road
    (212, 218, 30), # solid yellow line
    (118, 103, 122), # pavement_Sidewalk
    (225, 150, 119), # concrete_barrier
    (155, 104, 41), # fence
    (117, 82, 16), # police_vehicle  
    (126, 165, 140), # constuction_vehicle
    (169, 166, 90), # military_vehicle
    (160, 237, 109), # firebrigade 
    (74, 54, 62), # traditional truck
    (178, 9, 55), # container truck
    (200, 29, 177), # Hino_bus 
    (240, 9, 145), # traditional bus
    (255, 1, 77), # car
    (200, 165, 38), # van 
    (80, 216, 156), # mini_van
    (92, 83, 219), # rickshaw
    (253, 57, 87), # cart 
    (82, 35, 4), # pickup
    (196, 4, 86), # mini-pickup
    (163, 176, 81), # tanker 
    (82, 43, 112), # Ambulance
    (192, 99, 99), # motor_bike
    (55, 88, 15), # Tree
    (128, 222, 91), # VegetationMisc
    (196, 138, 200), # Signboard
    (240, 146, 5), # speedlimit
    (211, 203, 50), # Ad
    (181, 113, 130), # Directionboard
    (97, 132, 151), # distanceboard
    (167, 46, 47), # Pedestrian
    (120, 114, 104), # mountains
    (1, 37, 3), # grassy_road_divider
    (252, 2, 2), # bicycle
    ]

# all the classes that are present in the dataset
ALL_CLASSES = ['sky', 'bridge', 'building', 'wall', 'column_pole', 'trafficcone', 'electric_pole', 'street light', 'trafic light', 'pavedroad', 'unpaved_road', 'solid yellow line', 'pavement_sidewalk', 'concrete_barrier', 'fence', 'police_vehicle', 'constuction_vehicle', 'military_vehicle', 'firebrigade', 'traditional truck', 'container truck', 'hino_bus', 'traditional bus', 'car', 'van', 'mini_van', 'rickshaw', 'cart', 'pickup', 'mini-pickup', 'tanker', 'ambulance', 'motor_bike', 'tree', 'vegetationmisc', 'signboard', 'speedlimit' ,'ad', 'directionboard' ,'distanceboard' ,'pedestrian', 'mountains', 'grassy_road_divider', 'bicycle']

"""
This (`class_values`) assigns a specific class label to the each of the classes.
For example, `road=0`, `background=1`, and so on.
"""
class_values = [ALL_CLASSES.index(cls.lower()) for cls in config.CLASSES_TO_TRAIN]

class TensorboardWriter():
    def __init__(self):
        super(TensorboardWriter, self).__init__()
    # initilaize `SummaryWriter()`
        self.writer = SummaryWriter('runs/')
    def tensorboard_writer(self, loss, mIoU, pix_acc, iterations, phase=None):
        if phase == 'train':
            self.writer.add_scalar('Train Loss', loss, iterations)
            self.writer.add_scalar('Train mIoU', mIoU, iterations)
            self.writer.add_scalar('Train Pixel Acc', pix_acc, iterations)
        if phase == 'valid':
            self.writer.add_scalar('Valid Loss', loss, iterations)
            self.writer.add_scalar('Valid mIoU', mIoU, iterations)
            self.writer.add_scalar('Valid Pixel Acc', pix_acc, iterations)

def get_label_mask(mask, class_values): 
    """
    This function encodes the pixels belonging to the same class
    in the image into the same label
    """
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for value in class_values:
        for ii, label in enumerate(label_colors_list):
            if value == label_colors_list.index(label):
                label = np.array(label)
                label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


def draw_seg_maps(data, output, epoch, i):
    """
    This function color codes the segmentation maps that is generated while
    validating. THIS IS NOT TO BE CALLED FOR SINGLE IMAGE TESTING
    """
    alpha = 0.6 # how much transparency
    beta = 1 - alpha # alpht + beta should be 1
    gamma = 0 # contrast

    seg_map = output[0] # use only one output from the batch
    seg_map = torch.argmax(seg_map.squeeze(), dim=0).detach().cpu().numpy()

    image = data[0]
    image = np.array(image.cpu())
    image = np.transpose(image, (1, 2, 0))
    # unnormalize the image (important step)
    mean = np.array([0.45734706, 0.43338275, 0.40058118])
    std = np.array([0.23965294, 0.23532275, 0.2398498])
    image = std * image + mean
    image = np.array(image, dtype=np.float32)
    image = image * 255 # else OpenCV will save black image


    red_map = np.zeros_like(seg_map).astype(np.uint8)
    green_map = np.zeros_like(seg_map).astype(np.uint8)
    blue_map = np.zeros_like(seg_map).astype(np.uint8)
    
    for label_num in range(0, len(label_colors_list)):
        if label_num in class_values:
            idx = seg_map == label_num
            red_map[idx] = np.array(label_colors_list)[label_num, 0]
            green_map[idx] = np.array(label_colors_list)[label_num, 1]
            blue_map[idx] = np.array(label_colors_list)[label_num, 2]
        
    rgb = np.stack([red_map, green_map, blue_map], axis=2)
    rgb = np.array(rgb, dtype=np.float32)
    # convert color to BGR format for OpenCV
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.addWeighted(rgb, alpha, image, beta, gamma, image)
    cv2.imwrite(f"train_seg_maps/e{epoch}_b{i}.jpg", image)

def draw_test_segmentation_map(outputs):
    """
    This function will apply color mask as per the output that we
    get when executing `test.py` or `test_vid.py` on a single image 
    or a video. NOT TO BE USED WHILE TRAINING OR VALIDATING.
    """
    labels = torch.argmax(outputs.squeeze(), dim=0).detach().cpu().numpy()
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)
    
    for label_num in range(0, len(label_colors_list)):
        if label_num in class_values:
            idx = labels == label_num
            red_map[idx] = np.array(label_colors_list)[label_num, 0]
            green_map[idx] = np.array(label_colors_list)[label_num, 1]
            blue_map[idx] = np.array(label_colors_list)[label_num, 2]
        
    segmented_image = np.stack([red_map, green_map, blue_map], axis=2)
    return segmented_image

def image_overlay(image, segmented_image):
    """
    This function will apply an overlay of the output segmentation
    map on top of the orifinal input image. MAINLY TO BE USED WHEN
    EXECUTING `test.py` or `test_vid.py`.
    """
    alpha = 0.6 # how much transparency to apply
    beta = 1 - alpha # alpha + beta should equal 1
    gamma = 0 # scalar added to each sum
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    cv2.addWeighted(segmented_image, alpha, image, beta, gamma, image)
    return image

def visualize_from_dataloader(data_loader): 
    """
    Helper function to visualzie the data from 
    dataloaders. Only executes if `DEBUG` is `True` in
    `config.py`
    """
    data = iter(data_loader)   
    images, labels = data.next()
    image = images[1]
    # image = np.array(image, dtype='uint8') # use this if not normalizing, which is probably never
    image = np.array(image)
    image = np.transpose(image, (1, 2, 0))
    mean = np.array([0.45734706, 0.43338275, 0.40058118])
    std = np.array([0.23965294, 0.23532275, 0.2398498])
    image = std * image + mean
    image = np.array(image, dtype=np.float32)
    label = labels[1]
    images = [image, label.squeeze()]
    for i, image in enumerate(images):
        plt.subplot(1, 2, i+1)
        plt.imshow(image)
    plt.show()

def visualize_from_path(image_path, seg_path):
    """
    Helper function to visualize image and segmentation maps after
    reading from the images from path.
    Only executes if `DEBUG` is `True` in
    `config.py`
    """
    train_sample_img = cv2.imread(image_path[0])
    train_sample_img = cv2.cvtColor(train_sample_img, cv2.COLOR_BGR2RGB)
    train_sample_seg = cv2.imread(seg_path[0])
    train_sample_seg = cv2.cvtColor(train_sample_seg, cv2.COLOR_BGR2RGB)
    images = [train_sample_img, train_sample_seg]
    for i, image in enumerate(images):
        plt.subplot(1, 2, i+1)
        plt.imshow(image)
    plt.show()

def save_model_dict(model, epoch, optimizer, 
                    criterion, valid_iters, train_iters):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
            'valid_iters': valid_iters, 
            'train_iters': train_iters, 
            }, f"model.pth")
