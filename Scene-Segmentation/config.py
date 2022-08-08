EPOCHS = 50 # Number of the epochs
SAVE_EVERY = 5 # after how many epochs to save a checkpoint
LOG_EVERY = 1 #  log training and validation metrics every `LOG_EVERY` epochs
BATCH_SIZE = 2 
DEVICE = 'cuda'  
LR = 0.0001
ROOT_PATH = 'path to dataset directory'

CLASSES_TO_TRAIN = ['sky', 'bridge', 'building', 'wall', 'column_pole', 'trafficcone', 'electric_pole', 'street light', 'trafic light', 'pavedroad', 'unpaved_road', 'solid yellow line', 'pavement_sidewalk', 'concrete_barrier', 'fence', 'police_vehicle', 'constuction_vehicle', 'military_vehicle', 'firebrigade', 'traditional truck', 'container truck', 'hino_bus', 'traditional bus', 'car', 'van', 'mini_van', 'rickshaw', 'cart', 'pickup', 'mini-pickup', 'tanker', 'ambulance', 'motor_bike', 'tree', 'vegetationmisc', 'signboard', 'speedlimit' ,'ad', 'directionboard' ,'distanceboard' ,'pedestrian', 'mountains', 'grassy_road_divider', 'bicycle']

#CLASSES_TO_TRAIN = ['sky', 'bridge', 'building', 'wall', 'electric_pole', 'street light', 'pavedroad', 'unpaved_road', 'solid yellow line', 'pavement_sidewalk', 'concrete_barrier', 'fence', 'traditional truck', 'container truck', 'hino_bus', 'traditional bus', 'car', 'van', 'mini_van', 'rickshaw', 'pickup', 'mini-pickup', 'motor_bike', 'tree', 'vegetationmisc', 'signboard', 'speedlimit', 'directionboard' ,'distanceboard' ,'pedestrian', 'mountains', 'grassy_road_divider']

# DEBUG for visualizations
DEBUG = True