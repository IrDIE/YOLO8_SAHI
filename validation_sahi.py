from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG

from utils import *
import warnings


def get_category_mapping():
    # here should be all classes model was learning
    category_mapping = {
    '0': 'person',
    '1': 'bicycle',
    '2': 'car',
    '3': 'motorcycle',
    '4': 'airplane',
    '5': 'bus',
    '6': 'train',
    '7': 'truck',
    '8': 'boat',
    '9': 'traffic light',
    '10': 'fire hydrant',
    '11': 'stop sign',
    '12': 'parking meter',
    '13': 'bench',
    '14': 'bird',
    '15': 'cat',
    '16': 'dog',
    '17': 'horse',
    '18': 'sheep',
    '19': 'cow',
    '20': 'elephant',
    '21': 'bear',
    '22': 'zebra',
    '23': 'giraffe',
    '24': 'backpack',
    '25': 'umbrella',
    '26': 'handbag',
    '27': 'tie',
    '28': 'suitcase',
    '29': 'frisbee',
    '30': 'skis',
    '31': 'snowboard',
    '32': 'sports ball',
    '33': 'kite',
    '34': 'baseball bat',
    '35': 'baseball glove',
    '36': 'skateboard',
    '37': 'surfboard',
    '38': 'tennis racket',
    '39': 'bottle',
    '40': 'wine glass',
    '41': 'cup',
    '42': 'fork',
    '43': 'knife',
    '44': 'spoon',
    '45': 'bowl',
    '46': 'banana',
    '47': 'apple',
    '48': 'sandwich',
    '49': 'orange',
    '50': 'broccoli',
    '51': 'carrot',
    '52': 'hot dog',
    '53': 'pizza',
    '54': 'donut',
    '55': 'cake',
    '56': 'chair',
    '57': 'couch',
    '58': 'potted plant',
    '59': 'bed',
    '60': 'dining table',
    '61': 'toilet',
    '62': 'tv',
    '63': 'laptop',
    '64': 'mouse',
    '65': 'remote',
    '66': 'keyboard',
    '67': 'cell phone',
    '68': 'microwave',
    '69': 'oven',
    '70': 'toaster',
    '71': 'sink',
    '72': 'refrigerator',
    '73': 'book',
    '74': 'clock',
    '75': 'vase',
    '76': 'scissors',
    '77': 'teddy bear',
    '78': 'hair drier',
    '79': 'toothbrush'
}
    return category_mapping

def run_basic_validation(pt_model, yaml_datapath, args):
    validator = compile_validator(args=args, pt_modelpath=pt_model, yaml_datapath=yaml_datapath,
                                  save_dir=Path('./sahi/res_no_sahi/'))
    LOGGER.info('Starting validation')
    validator()

def run_sahi_validation(pt_model, yaml_datapath, args, imgsz, device):
    validator_sahi = compile_validator(args=args, pt_modelpath=pt_model, yaml_datapath=yaml_datapath,
                                       save_dir=Path('./sahi/res_SAHI/'), imgsz=imgsz, sahi=True)
    category_mapping = get_category_mapping()  # rewrite that function with your classes
    sahi_model = get_sahi_model(pt_model, category_mapping=category_mapping)
    sahi_model.model.to(device)
    LOGGER.info('Starting SAHI validation')
    validator_sahi(sahi_model=sahi_model)

def main():
    warnings.filterwarnings("ignore")
    # here set your parameters
    pt_model = './yolov8m.pt'
    yaml_datapath = './sahi_data.yaml'
    imgsz = 640*2

    # defaults params
    args = get_cfg(cfg=DEFAULT_CFG)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    LOGGER.info(f'device === {device}')

    # run_sahi_validation or run_basic_validation
    #run_basic_validation(pt_model, yaml_datapath, args)
    run_sahi_validation(pt_model, yaml_datapath, args, imgsz, device)

if __name__ == '__main__':
    main()
