# SAHI inference with YOLOv8 :dizzy:  boost your small object detection performance 

This repo contain not only SAHI inference implementation but also evaluation of results with mAp50, ... (standart metrics)

You will understand if SAHI inference help in your specific case

### What in this repo:
* SAHI inference + EVALUATION of results with basic yolo8 metrics 
  * output example with basic validation on 2 images:
    * ```python
          val: Scanning C:\Users\irady\GitHub\YOLO8_SAHI\yolo_dataset\labels.cache.
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 
                       all          2         11      0.987      0.545       0.57      0.455
                    person          2         11      0.987      0.545       0.57      0.455

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
        
        ```
        
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
      
## How to use:

* git clone
* in `validation_sahi.main()` run `run_sahi_validation()` or `run_basic_validation()`
* in `.main()` change paths for your .pt and .yaml