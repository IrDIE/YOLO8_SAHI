from utils import *


def run_basic_validation(pt_model, yaml_datapath, args, imgsz):
    validator = compile_validator(args=args, pt_modelpath=pt_model, imgsz=imgsz, yaml_datapath=yaml_datapath,
                                  save_dir=Path('./sahi/validation/res_no_sahi/'))
    LOGGER.info('Starting basic validation')
    val_res = validator()
    return val_res

def run_sahi_validation(pt_model, yaml_datapath, args, imgsz):
    validator_sahi = compile_validator(args=args, pt_modelpath=pt_model, yaml_datapath=yaml_datapath,
                                       save_dir=Path('./sahi/validation/res_SAHI/'), imgsz=imgsz, sahi=True)
    LOGGER.info('Starting SAHI validation')
    val_res = validator_sahi()
    return val_res



