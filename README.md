# SAHI inference with YOLOv8 :dizzy:  boost your small object detection performance 

### What is SAHI (Slicing Aided Hyper Inference) - https://docs.ultralytics.com/guides/sahi-tiled-inference/
### What in this repo:
This repo contain not only SAHI inference implementation but also evaluation of results with mAp50, ... (standart metrics)

You will understand if SAHI inference help in your specific case

| **SAHI** 	                            | **No SAHI** 	                   |
|---------------------------------------|---------------------------------|
| 	![pred_sahi.jpg](https://github.com/IrDIE/YOLO8_SAHI/blob/master/readme_imgs/pred_sahi.jpg)                                     | 	    ![pred_no_sahi.jpg](https://github.com/IrDIE/YOLO8_SAHI/blob/master/readme_imgs/pred_no_sahi.jpg)                           |
| more cars far away detected         	 | standart detections           	 |

## Run output:
* SAHI inference + EVALUATION of results with basic yolo8 metrics 
  * output example with basic validation on 2 images:
    ```python
          val: Scanning C:\Users\irady\GitHub\YOLO8_SAHI\yolo_dataset\labels.cache.
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 
                       all          2         11      0.987      0.545       0.57      0.455
                    person          2         11      0.987      0.545       0.57      0.455
    Speed: 1.5ms preprocess, 329.6ms inference, 0.0ms loss, 4.5ms postprocess per image

        ```
* output example with **SAHI validation** on 2 images:
    * ```python
      val: Scanning C:\Users\irady\GitHub\YOLO8_SAHI\yolo_dataset\labels.cache.
               Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 
      Performing prediction on 9 number of slices.
      Performing prediction on 9 number of slices.
               Class     Images  Instances      Box(P          R      mAP50  mAP50-95):
                 all          2         11          1      0.545      0.773      0.628
              person          2         11          1      0.545      0.773      0.628
      Speed: 7.5ms preprocess, 0.0ms inference, 0.0ms loss, 0.0ms postprocess per image
        
        ```
* and also check `sahi/` folder - there all validation plots will be saved  

 
## How to use:

* git clone
* in `utils.get_category_mapping() change returned dictionary for your classes`
* in `validation_sahi.main()` run `run_sahi_validation()` or `run_basic_validation()`
* in `validation_sahi.main()` change paths for your .pt and .yaml, and set desired input imgsz 
* also you can update size  of sliding window (`slice_width` and `slice_height`) in `utils.sahi_predict()` (640 default)



## What you need:
* your .pt model file
* validation dataset in yolo standart format
* .yaml file for dataset
  * ```angular2html
    #file sahi_data.yaml
    path: ../YOLO8_SAHI/yolo_dataset/ # dataset root dir

    train: images # train images (relative to 'path') 128 images
    val: images # val images (relative to 'path') 128 images
    
    # Classes
    names:
      0: people
    ```
     