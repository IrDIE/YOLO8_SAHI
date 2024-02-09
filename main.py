import warnings
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from validation_sahi import run_sahi_validation, run_basic_validation, LOGGER
from inference_sahi import run_sahi_prediction, run_basic_prediction
import torch

def main():
    warnings.filterwarnings("ignore")
    # here set your parameters
    pt_model = './best.pt'
    yaml_datapath = './sahi_data.yaml' # for validation
    imgsz = 640*2 # for sahi validation or prediction
    predict_source = './yolo_dataset/images'
    video1_source = './1.mp4'


    # defaults params
    args = get_cfg(cfg=DEFAULT_CFG)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    LOGGER.info(f'DEVICE ===>> {device}')

    # VALIDATION
    run_basic_validation(pt_model, yaml_datapath, args)
    run_sahi_validation(pt_model, yaml_datapath, args, imgsz, device)

    # PREDICTION (INFERENCE)
    run_sahi_prediction(args, pt_model, source = video1_source, imgsz = imgsz , device = device)
    run_basic_prediction(pt_model=pt_model, args = args, source=video1_source)

if __name__ == '__main__':
    main()