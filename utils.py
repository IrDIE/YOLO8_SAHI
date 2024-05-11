from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import TQDM, callbacks, colorstr, emojis
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import de_parallel, select_device
from ultralytics.models.yolo.detect import DetectionValidator, DetectionPredictor
from sahi.predict import get_sliced_prediction
from torchvision.transforms import v2
from sahi import AutoDetectionModel
import torch
from pathlib import Path
import numpy as np
from ultralytics.utils.checks import check_imgsz, check_imshow
from sahi.slicing import get_slice_bboxes
import threading
from ultralytics.cfg import get_save_dir
from ultralytics.utils import LOGGER, ops
from ultralytics.engine.results import Results
import cv2
import json

SLICE_H = 320 # any multiple by 32 number
SLICE_W = 320
OVERLAP_HEIGHT_RATIO = 0.2
OVERLAP_WIDTH_RATIO = 0.2
VERBOSE_SAHI = 0
"""
VERBOSE_SAHI: int
            0: no print
            1: print number of slices (default)
            2: print number of slices and slice/prediction durations
"""


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

def get_sahi_model(pt_path, model_type='yolov8', device = 'cpu'):
    category_mapping = get_category_mapping() # rewrite that function with your classes
    detection_model = AutoDetectionModel.from_pretrained(model_type = model_type, model_path = pt_path, \
                                                         device = device, category_mapping = category_mapping)
    detection_model.model.to(device)
    return detection_model


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

    def compute_window_shape(self, orig_shape, window_shape=[SLICE_H, SLICE_W]):
        """
        - we need compile model in advance with appropriate image size (yolo modela require imgsize be multiple 32)
        - that function fix cases when inpur original resolution lower than required window-sahi size (for example 720 * 1080 == Yolo can't process 720 imgsize)
        :param orig_shape: (720, 1280) tuple
        :param window_shape: hyperparameter
        :return: new shape model can be compiled with
        """
        h_orig, w_orig = orig_shape
        if window_shape[0] > h_orig: window_shape[0] = (h_orig | 31) - 31
        if window_shape[1] > w_orig: window_shape[1] = (w_orig | 31) - 31
        return window_shape
    def sahi_inference(
            self,
            im,
            slice_height=SLICE_H,
            slice_width=SLICE_W,
            overlap_height_ratio=OVERLAP_HEIGHT_RATIO,
            overlap_width_ratio=OVERLAP_WIDTH_RATIO,
            *args,
            **kwargs,
    ):
        """
        detection_model : compiled from def get_sahi_model()
        image_batch : torch.Size([10, 3, 1280, 1280])
        run_usual : if run usual inference for all picture (not sliced) above sahi prediction
        """
        btch, ch, image_height, image_width = im.shape
        # 1 - get sliced image coordinates
        slice_bboxes = get_slice_bboxes(
            image_height=image_height,
            image_width=image_width,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
        )
        # 2 - in loop do usual inference for each slice -> get predictions for each slice, append it to result list
        # check later gpu availability
        preds_all_slice_shifted = []
        for x1, y1, x2, y2 in slice_bboxes:
            # take all slices as batch = im and do inference on batch
            preds = self.model(
                im[:, :, y1:y2, x1:x2], *args, **kwargs
            )  # im here is batch !!
            if isinstance(
                    preds, (list, tuple)
            ):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
                preds_take = preds[0]
            else:
                preds_take = preds
            preds_t = preds_take.transpose(-1, -2)
            prediction_slice = preds_t[..., :4]  # in xywh
            # shift prediction regarding to slice coordinates. got pregictionn regarding scale of original image
            prediction_slice[:, :, 0] += x1  # shift x center
            prediction_slice[:, :, 1] += y1  # shift y center
            preds_t[..., :4] = prediction_slice  # shift preds
            if isinstance(
                    preds, (list, tuple)
            ):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
                preds[0] = preds_t.transpose(-1, -2)
                preds_all_slice_shifted.append(preds[0])
            else:
                preds = preds_t.transpose(-1, -2)
                preds_all_slice_shifted.append(preds)
        preds_all_slice_shifted_t = torch.cat((preds_all_slice_shifted), dim=2)
        return preds_all_slice_shifted_t
    def __call__(self, trainer=None, model=None):
        """Supports validation of a pre-trained model if passed or a model being trained if trainer is passed (trainer
        gets priority).
        """
        self.training = trainer is not None

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

            slice_h, slice_w = self.compute_window_shape((imgsz,imgsz), window_shape=[SLICE_H, SLICE_W])

            model.warmup(imgsz=(1 if pt else self.args.batch, 3, slice_h, slice_w))  # warmup

        self.run_callbacks('on_val_start')
        dt = Profile(), Profile(), Profile(), Profile()
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        # bar is different

        self.init_metrics(de_parallel(model))
        self.jdict = []  # empty before each val
        self.model = model
        for batch_i, batch in enumerate(bar):
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)

            self.run_callbacks('on_val_batch_start')
            self.batch_i = batch_i
            # Preprocess

            with dt[0]:

                batch_sahi = self.preprocess(batch)
                im = v2.Resize(size=(slice_h, slice_w))(batch_sahi["img"])
                from_shape = im.shape[2:]
                to_shape = batch_sahi["img"].shape[2:]
            # Inference
            with dt[1]:
                # -------------------
                if self.usual_inference:
                    preds = self.model(im, augment=augment)
                    w_gain, h_gain = (
                        to_shape[1] / from_shape[1],
                        to_shape[0] / from_shape[0],
                    )
                    if isinstance(preds, (list, tuple)):
                        transposed = preds[0].transpose(
                            -1, -2
                        )  # preds in xywh for torch.Size([640, 640])
                        for box in transposed[..., :4]:
                            box[:, 0] *= w_gain
                            box[:, 1] *= h_gain
                            box[:, 2] *= w_gain
                            box[:, 3] *= h_gain
                        scaled_xywh = transposed.transpose(-1, -2)
                        preds[0] = scaled_xywh
                    else:
                        transposed = preds.transpose(
                            -1, -2
                        )  # preds in xywh for torch.Size([640, 640])
                        for box in transposed[..., :4]:
                            box[:, 0] *= w_gain
                            box[:, 1] *= h_gain
                            box[:, 2] *= w_gain
                            box[:, 3] *= h_gain
                        scaled_xywh = transposed.transpose(-1, -2)
                        preds = scaled_xywh
                if self.sahi:
                    preds_sahi = self.sahi_inference(im=batch_sahi["img"], slice_height=slice_h, slice_width=slice_w, augment=False)
                if self.sahi and self.usual_inference:
                    if isinstance(preds, (list, tuple)):
                        preds[0] = torch.cat((preds[0], preds_sahi), dim=2)
                    else:
                        preds = torch.cat((preds, preds_sahi), dim=2)

            # Loss
            with dt[2]:
                if self.training:
                    self.loss += model.loss(batch, preds)[1]
            # Postprocess
            with dt[3]:
                if not self.usual_inference:
                    preds = preds_sahi
                preds = self.postprocess(preds, to_shape, batch["img"])
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
    def postprocess(self, preds, from_shape, orig_imgs):
        """Post-processes predictions and returns a list of Results objects.
        from_shape : img.shape[2:]
        """
        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )


class DetectionPredictor_SAHI(DetectionPredictor):

    def __init__(self, args, overrides=None, _callbacks=None):
        """
        Initializes the BasePredictor class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = args
        self.save_dir = get_save_dir(self.args)
        if self.args.conf is None:
            self.args.conf = 0.25  # default conf=0.25
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        # Usable if setup is done
        self.model = None
        self.data = self.args.data  # data_dict
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_path, self.vid_writer, self.vid_frame = None, None, None
        self.plotted_img = None
        self.data_path = None
        self.source_type = None
        self.batch = None
        self.results = None
        self.transforms = None
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        self.txt_path = None
        self._lock = threading.Lock()  # for automatic thread-safe inference
        callbacks.add_integration_callbacks(self)

    # def __call__(self, source=None, model=None, stream=False, *args, **kwargs):
    #     return list(self.stream_inference(source, model, *args, **kwargs))  # merge list of Result into one

    def __call__(self, source=None, model=None, stream=False, *args, **kwargs):

        self.stream = stream
        if stream:
            return self.stream_inference(source, model, *args, **kwargs)
        else:
            return list(
                self.stream_inference(source, model,*args, **kwargs)
            )

    def compute_window_shape(self, orig_shape, window_shape=[SLICE_H, SLICE_W]):
        """
        - we need compile model in advance with appropriate image size (yolo modela require imgsize be multiple 32)
        - that function fix cases when inpur original resolution lower than required window-sahi size (for example 720 * 1080 == Yolo can't process 720 imgsize)
        :param orig_shape: (720, 1280) tuple
        :param window_shape: hyperparameter
        :return: new shape model can be compiled with
        """
        h_orig, w_orig = orig_shape
        if window_shape[0] > h_orig: window_shape[0] = (h_orig | 31) - 31
        if window_shape[1] > w_orig: window_shape[1] = (w_orig | 31) - 31
        return window_shape
    def sahi_inference(
            self,
            im,
            slice_height=SLICE_H,
            slice_width=SLICE_W,
            overlap_height_ratio=OVERLAP_HEIGHT_RATIO,
            overlap_width_ratio=OVERLAP_WIDTH_RATIO,
            *args,
            **kwargs,
    ):
        """
        detection_model : compiled from def get_sahi_model()
        image_batch : torch.Size([10, 3, 1280, 1280])
        run_usual : if run usual inference for all picture (not sliced) above sahi prediction
        """
        btch, ch, image_height, image_width = im.shape
        # 1 - get sliced image coordinates
        slice_bboxes = get_slice_bboxes(
            image_height=image_height,
            image_width=image_width,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
        )
        # 2 - in loop do usual inference for each slice -> get predictions for each slice, append it to result list
        # check later gpu availability
        preds_all_slice_shifted = []
        for x1, y1, x2, y2 in slice_bboxes:
            # take all slices as batch = im and do inference on batch
            preds = self.inference(
                im[:, :, y1:y2, x1:x2], *args, **kwargs
            )  # im here is batch !!
            if isinstance(
                    preds, (list, tuple)
            ):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
                preds_take = preds[0]
            else:
                preds_take = preds
            preds_t = preds_take.transpose(-1, -2)
            prediction_slice = preds_t[..., :4]  # in xywh
            # shift prediction regarding to slice coordinates. got pregictionn regarding scale of original image
            prediction_slice[:, :, 0] += x1  # shift x center
            prediction_slice[:, :, 1] += y1  # shift y center
            preds_t[..., :4] = prediction_slice  # shift preds
            if isinstance(
                    preds, (list, tuple)
            ):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
                preds[0] = preds_t.transpose(-1, -2)
                preds_all_slice_shifted.append(preds[0])
            else:
                preds = preds_t.transpose(-1, -2)
                preds_all_slice_shifted.append(preds)
        preds_all_slice_shifted_t = torch.cat((preds_all_slice_shifted), dim=2)
        return preds_all_slice_shifted_t

    def postprocess(self, preds, from_shape, orig_imgs):
        """Post-processes predictions and returns a list of Results objects.
        from_shape : img.shape[2:]
        """
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )
        if not isinstance(
                orig_imgs, list
        ):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(from_shape, pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            results.append(
                Results(orig_img, path=img_path, names=self.model.names, boxes=pred)
            )
        return results
    def stream_inference(self, source=None, model=None, *args, **kwargs):
        """Streams real-time inference on camera feed and saves results to file."""

        # Setup model
        if not self.model:
            self.setup_model(model)
        self.nc = len(self.model.names)
        self.stride = self.model.stride
        with self._lock:  # for thread-safe inference
            # Setup source every time predict is called
            self.setup_source(source if source is not None else self.args.source)

            self.imgsz = self.args.sahi_imgsz
            if self.args.dynamic_input : LOGGER.warning("\nWARNING! Set dynamic_input = True : using original image size for each prediction that can take more time.\n")

            # Check if save_dir/ label file exists
            if self.args.save or self.args.save_txt:
                (
                    self.save_dir / "labels" if self.args.save_txt else self.save_dir
                ).mkdir(parents=True, exist_ok=True)
            # Warmup model

            self.seen, self.windows, self.batch = 0, [], None
            profilers = (Profile(), Profile(), Profile())
            self.run_callbacks("on_predict_start")
            for batch in self.dataset:
                # self.run_callbacks("on_predict_batch_start")
                self.batch = batch
                path, im0s, vid_cap, s = batch
                input_image_shape = np.shape(im0s[0])[:2]
                if self.args.dynamic_input:
                    self.imgsz = input_image_shape # HW
                    LOGGER.info(f'\n*** set self.imgsz as original image shape = {self.imgsz}')

                slice_h, slice_w = self.compute_window_shape(self.imgsz, window_shape=[SLICE_H, SLICE_W])

                if not self.done_warmup:
                    self.model.warmup(
                        imgsz=(
                            1 if self.model.pt or self.model.triton else self.dataset.bs,
                            3,
                            slice_h,
                            slice_w,
                        )
                    )
                    self.done_warmup = True
                # Preprocess
                with profilers[0]:
                    im_sahi = self.preprocess(im0s)  # CHW
                    im = v2.Resize(size=(slice_h, slice_w))(im_sahi)
                    from_shape = im.shape[2:]
                    to_shape = im_sahi.shape[2:]
                # Inference
                with profilers[1]:
                    if self.args.usual_inference:
                        preds = self.inference(im, *args, **kwargs)
                        w_gain, h_gain = (
                            to_shape[1] / from_shape[1],
                            to_shape[0] / from_shape[0],
                        )
                        if isinstance(preds, (list, tuple)):
                            transposed = preds[0].transpose(
                                -1, -2
                            )  # preds in xywh for torch.Size([640, 640])
                            for box in transposed[..., :4]:
                                box[:, 0] *= w_gain
                                box[:, 1] *= h_gain
                                box[:, 2] *= w_gain
                                box[:, 3] *= h_gain
                            scaled_xywh = transposed.transpose(-1, -2)
                            preds[0] = scaled_xywh
                        else:
                            transposed = preds.transpose(
                                -1, -2
                            )  # preds in xywh for torch.Size([640, 640])
                            for box in transposed[..., :4]:
                                box[:, 0] *= w_gain
                                box[:, 1] *= h_gain
                                box[:, 2] *= w_gain
                                box[:, 3] *= h_gain
                            scaled_xywh = transposed.transpose(-1, -2)
                            preds = scaled_xywh
                    if self.args.sahi:
                        preds_sahi = self.sahi_inference(im=im_sahi, slice_height=slice_h, slice_width=slice_w)
                    if self.args.sahi and self.args.usual_inference:
                        if isinstance(preds, (list, tuple)):
                            preds[0] = torch.cat((preds[0], preds_sahi), dim=2)
                        else:
                            preds = torch.cat((preds, preds_sahi), dim=2)
                # Postprocess
                with profilers[2]:
                    if not self.args.usual_inference:
                        preds = preds_sahi
                    self.results = self.postprocess(preds, to_shape, im0s)

                self.run_callbacks("on_predict_postprocess_end")

                # Visualize, save, write results
                n = len(im0s)
                for i in range(n):
                    self.seen += 1
                    self.results[i].speed = {
                        "preprocess": profilers[0].dt * 1e3 / n,
                        "inference": profilers[1].dt * 1e3 / n,
                        "postprocess": profilers[2].dt * 1e3 / n,
                    }
                    p, im0 = path[i], None if self.source_type.tensor else im0s[i].copy()
                    p = Path(p)

                    if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                        s += self.write_results(i, self.results, (p, im, im0))
                    if self.args.save or self.args.save_txt:
                        self.results[i].save_dir = self.save_dir.__str__()
                    if self.args.show and self.plotted_img is not None:
                        self.show(p)
                    if self.args.save and self.plotted_img is not None:
                        self.save_preds(vid_cap, i, str(self.save_dir / p.name))

                self.run_callbacks("on_predict_batch_end")
                yield from self.results

                # Print time (inference-only)
                if self.args.verbose:
                    LOGGER.info(f"{s}{profilers[1].dt * 1E3:.1f}ms")

        # Release assets
        if isinstance(self.vid_writer[-1], cv2.VideoWriter):
            self.vid_writer[-1].release()  # release final video writer

        # Print results
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1e3 for x in profilers)  # speeds per image
            LOGGER.info(
                f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape "
                f"{(1, 3, *im.shape[2:])}" % t
            )
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob("labels/*.txt")))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ""
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")

        self.run_callbacks("on_predict_end")

def compile_validator(args, pt_modelpath, yaml_datapath, save_dir, imgsz = None, sahi = False, usual_inference=True):
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
    validator.sahi = sahi
    validator.usual_inference = usual_inference
    validator.nc = len(model.names)
    validator.metrics.names = validator.names
    validator.metrics.plot = validator.args.plots
    validator.data = check_det_dataset(args.data)
    validator.training = False
    validator.stride = model.stride
    LOGGER.info(f'\nValidator {"SAHI" if sahi else ""} compiled successfully!')
    return validator

def compile_predictor(args, pt_modelpath, save_dir, iou_thr = 0.5, conf = 0.5, imgsz = None, sahi = False, usual_inference = True):
    args.iou = iou_thr
    args.conf = conf
    args.model = pt_modelpath
    args.sahi = sahi
    args.usual_inference = usual_inference
    args.sahi_imgsz = imgsz
    args.dynamic_input = True if imgsz is None else False

    predictor = DetectionPredictor_SAHI(args = args) if sahi else DetectionPredictor()
    # predictor.model = pt_modelpath
    predictor.save_dir = save_dir
    # predictor.device = device
    # predictor.imgsz = imgsz
    # predictor.args.imgsz

    return predictor

def sahi_predict(detection_model, image_batch, slice_height = SLICE_H, slice_width = SLICE_W, \
                 overlap_height_ratio = OVERLAP_HEIGHT_RATIO, overlap_width_ratio = OVERLAP_WIDTH_RATIO):
    """
    detection_model : compiled from def get_sahi_model()
    image_batch : torch.Size([10, 3, 1280, 1280])
    """
    device_orig = detection_model.model.device
    batch_result = []
    for image in image_batch:
        box_annot = np.empty((0, 6)) #
        if isinstance(image,  torch.Tensor):
            image = image.cpu().numpy() * 255
            image = np.transpose(image, (1, 2, 0)).astype(np.uint8) # in CHW

        result = get_sliced_prediction(
            image,
            detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            verbose = VERBOSE_SAHI
        )
        for img_box in result.object_prediction_list:
            x1, y1, x2, y2 = img_box.bbox.to_xyxy()
            confidence = img_box.score.value
            cls = img_box.category.id
            box_annot = np.concatenate((box_annot, [[x1, y1, x2, y2, confidence, cls]]))
        batch_result.append(torch.tensor(box_annot).to(device_orig))
    return batch_result
