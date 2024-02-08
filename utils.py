from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import de_parallel, select_device, smart_inference_mode
from ultralytics.models.yolo.detect import DetectionValidator
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel
import torch
from pathlib import Path
import numpy as np


class DetectionValidator_SAHI(DetectionValidator):
    """
        A class extending the DetectionValidator class for sahi inference for detection model.

        Example:
            ```python
            args = dict(model='yolov8n.pt', data='coco8.yaml')
            validator = DetectionValidator_SAHI(args=args)
            validator()
            ```
        """
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize detection model with necessary variables and settings."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)


    def __call__(self, trainer=None, model=None, sahi_model=None):
        """Supports validation of a pre-trained model if passed or a model being trained if trainer is passed (trainer
        gets priority).
        """
        self.training = trainer is not None
        self.sahi_model = sahi_model
        augment = self.args.augment and (not self.training)
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            self.args.half = self.device.type != 'cpu'  # force FP16 val during training
            model = trainer.ema.ema or trainer.model
            model = model.half() if self.args.half else model.float()
            # self.model = model
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
        else:
            callbacks.add_integration_callbacks(self)
            model = AutoBackend(model or self.args.model,
                                device=select_device(self.args.device, self.args.batch),
                                dnn=self.args.dnn,
                                data=self.args.data,
                                fp16=self.args.half)
            # self.model = model
            self.device = model.device  # update device
            self.args.half = model.fp16  # update half
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if engine:
                self.args.batch = model.batch_size
            elif not pt and not jit:
                self.args.batch = 1  # export.py models default to batch-size 1
                LOGGER.info(f'Forcing batch=1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

            if isinstance(self.args.data, str) and self.args.data.split('.')[-1] in ('yaml', 'yml'):
                self.data = check_det_dataset(self.args.data)
            elif self.args.task == 'classify':
                self.data = check_cls_dataset(self.args.data, split=self.args.split)
            else:
                raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

            if self.device.type in ('cpu', 'mps'):
                self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
            if not pt:
                self.args.rect = False
            self.stride = model.stride  # used in get_dataloader() for padding
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)
            model.eval()
            model.warmup(imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz))  # warmup
        self.run_callbacks('on_val_start')
        dt = Profile(), Profile(), Profile(), Profile(), Profile()
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        # bar is different

        self.init_metrics(de_parallel(model))
        self.jdict = []  # empty before each val
        for batch_i, batch in enumerate(bar):
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)

            self.run_callbacks('on_val_batch_start')
            self.batch_i = batch_i
            # Preprocess

            with dt[0]:
                batch = self.preprocess(batch)
            # Inference
            with dt[1]:
                if self.sahi_model is None:
                    #LOGGER.info(f'\n batch[img] shape = {batch["img"].shape} ') # == batch[img] shape = torch.Size([3, 3, 640, 640])
                    preds = model(batch['img'], augment=augment)
            # SAHI INFERENCE
            with dt[4]:
                if self.sahi_model :
                    preds = sahi_predict(self.sahi_model, batch['img'])
            # Loss
            with dt[2]:
                if self.training:
                    self.loss += model.loss(batch, preds)[1]
            # Postprocess
            with dt[3]:
                preds = self.postprocess(preds) if self.sahi_model is None else preds
                #LOGGER.info(f'preds shape in Postprocess = {preds[1].shape}') #x1, y1, x2, y2, confidence, class : [ 84, 9, 639, 636, 0.89,16]
                #[tensor([[ 8.4056e+01,  9.2479e+00,  6.3970e+02,  6.3664e+02,  8.9213e-01,  1.6000e+01], ...
            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3:
                self.plot_predictions(batch, preds, batch_i)
            self.run_callbacks('on_val_batch_end')
        stats = self.get_stats()
        self.check_stats(stats)
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1E3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks('on_val_end')
        if self.training:
            model.float()
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix='val')}
            return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
        else:
            LOGGER.info('Speed: %.1fms preprocess, %.1fms inference, %.1fms loss, %.1fms postprocess per image' %
                        tuple(self.speed.values()))
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / 'predictions.json'), 'w') as f:
                    LOGGER.info(f'Saving {f.name}...')
                    json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats)  # update stats
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
            return stats

def get_sahi_model(pt_path, category_mapping, model_type='yolov8', device = 'cpu'):
    detection_model = AutoDetectionModel.from_pretrained(model_type = model_type, model_path = pt_path, \
                                                         device = device, category_mapping = category_mapping)
    return detection_model

def compile_validator(args, pt_modelpath, yaml_datapath, save_dir = Path('./sahi/res/'), imgsz = 640, sahi = False):
    args.model = pt_modelpath
    args.data = yaml_datapath
    args.imgsz = imgsz

    validator = DetectionValidator_SAHI(args=args,save_dir=save_dir) if sahi else DetectionValidator(args=args,save_dir=save_dir)
    validator.is_coco = False

    model = AutoBackend(
        validator.args.model,
        device=select_device(validator.args.device, validator.args.batch),
        dnn=validator.args.dnn,
        data=validator.args.data,
        fp16=validator.args.half,
    )

    validator.names = model.names
    validator.nc = len(model.names)
    validator.metrics.names = validator.names
    validator.metrics.plot = validator.args.plots
    validator.data = check_det_dataset(args.data)
    validator.training = False
    validator.stride = model.stride
    LOGGER.info(f'\nValidator {"SAHI" if sahi else ""} compiled successfully!')
    return validator

def sahi_predict(detection_model, image_batch, slice_height = 640, slice_width = 640, overlap_height_ratio = 0.2, overlap_width_ratio = 0.2):
    """
    detection_model : compiled from def get_sahi_model()
    image_batch : torch.Size([10, 3, 1280, 1280])
    """
    device_orig = detection_model.model.device
    batch_result = []
    for image in image_batch:
        box_annot = np.empty((0, 6)) #
        image = image.cpu().numpy() * 255
        image = np.transpose(image, (1, 2, 0)).astype(np.uint8)

        result = get_sliced_prediction(
            image,
            detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio
        )
        for img_box in result.object_prediction_list:
            x1, y1, x2, y2 = img_box.bbox.to_xyxy()
            confidence = img_box.score.value
            cls = img_box.category.id
            box_annot = np.concatenate((box_annot, [[x1, y1, x2, y2, confidence, cls]]))
        batch_result.append(torch.tensor(box_annot).to(device_orig))
    return batch_result
