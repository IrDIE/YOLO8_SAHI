from utils import *


def run_basic_validation(pt_model, yaml_datapath, args):
    validator = compile_validator(args=args, pt_modelpath=pt_model, yaml_datapath=yaml_datapath,
                                  save_dir=Path('./sahi/validation/res_no_sahi/'))
    LOGGER.info('Starting basic validation')
    val_res = validator()
    return val_res

def run_sahi_validation(pt_model, yaml_datapath, args, imgsz, device):
    validator_sahi = compile_validator(args=args, pt_modelpath=pt_model, yaml_datapath=yaml_datapath,
                                       save_dir=Path('./sahi/validation/res_SAHI/'), imgsz=imgsz, sahi=True)
    sahi_model = get_sahi_model(pt_model, device = device)
    LOGGER.info('Starting SAHI validation')
    val_res = validator_sahi(sahi_model=sahi_model)
    return val_res



