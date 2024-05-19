"""
this function used only for ability to use original-sized images during validation

"""

from ultralytics.utils import LOGGER

import math
from copy import deepcopy
import cv2
import numpy as np
from pathlib import Path

def load_image(self, i, rect_mode=True):
    """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
    im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
    if im is None:  # not cached in RAM
        if fn.exists():  # load npy
            try:
                im = np.load(fn)
            except Exception as e:
                LOGGER.warning(f'{self.prefix}WARNING ⚠️ Removing corrupt *.npy image file {fn} due to: {e}')
                Path(fn).unlink(missing_ok=True)
                im = cv2.imread(f)  # BGR
        else:  # read image
            im = cv2.imread(f)  # BGR
        if im is None:
            raise FileNotFoundError(f'Image Not Found {f}')

        h0, w0 = im.shape[:2]  # orig hw
        # if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
        #     r = self.imgsz / max(h0, w0)  # ratio
        #     if r != 1:  # if sizes are not equal
        #         w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
        #         im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        # elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
        #     im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

        # Add to buffer if training with augmentations
        # if self.augment:
        #     self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
        #     self.buffer.append(i)
        #     if len(self.buffer) >= self.max_buffer_length:
        #         j = self.buffer.pop(0)
        #         self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

        return im, (h0, w0), im.shape[:2]

    return self.ims[i], self.im_hw0[i], self.im_hw[i]



def get_image_and_label(self, index):
    """Get and return label information from the dataset."""
    label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
    label.pop('shape', None)  # shape is for rect, remove it
    label['img'], label['ori_shape'], label['resized_shape'] = self.load_image(index)
    label['ratio_pad'] = (label['resized_shape'][0] / label['ori_shape'][0],
                          label['resized_shape'][1] / label['ori_shape'][1])  # for evaluation
    # if self.rect:
    label['rect_shape'] = label['ori_shape'] #self.batch_shapes[self.batch[index]]
    return self.update_labels_info(label)