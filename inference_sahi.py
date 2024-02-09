from utils import *
def run_sahi_prediction(args, pt_model, source, imgsz , device):
    predictor = compile_predictor(args, pt_modelpath=pt_model, save_dir = Path('./sahi/prediction/res_SAHI/'), sahi=True, imgsz = imgsz)
    sahi_model = get_sahi_model(pt_model,device=device)

    LOGGER.info('Starting SAHI prediction')
    result = predictor(sahi_model = sahi_model, source = source)
    return result


def run_basic_prediction(pt_model, args, source):
    predictor = compile_predictor(args, pt_modelpath = pt_model, save_dir = Path('./sahi/prediction/res/'))
    LOGGER.info('Starting basic prediction')
    result = predictor(model = pt_model, source = source)
    return result
