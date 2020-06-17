## Expected dataset structure for cityscapes:
```
cityscapes/
  gtFine/
    train/
      aachen/
        *_color.png, *_instanceIds.png, *_labelIds.png, *_polygons.json,
        *_labelTrainIds.png
      ...
    val/
    test/
    cityscapes_panoptic_train_trainId.json
    cityscapes_panoptic_val_trainId.json
    cityscapes_panoptic_train_trainId/
      *_panoptic.png
    cityscapes_panoptic_val_trainId/
      *_panoptic.png
  leftImg8bit/
    train/
    val/
    test/
```
Install cityscapes scripts by:
```
pip install git+https://github.com/mcordts/cityscapesScripts.git
```

Note:
- labelTrainIds.png are created by `python cityscapesscripts/preparation/createTrainIdLabelImgs.py`.  
- panoptic.png are created by `python cityscapesscripts/preparation/createPanopticImgs.py --use-train-id`.

## Expected dataset structure for COCO panoptic segmentation:

```
coco/
  annotations/
    instances_{train,val}2017.json
    panoptic_{train,val}2017.json
    panoptic_{train,val}2017_trainId.json
    panoptic_{train,val}2017/  # png annotations
  {train,val}2017/
    # image files that are mentioned in the corresponding json
```

Install panopticapi by:
```
pip install git+https://github.com/cocodataset/panopticapi.git
```

Note:
- panoptic_{train,val}2017_trainId.json are created by `python prepare_coco_panoptic_trainid.py`.  
