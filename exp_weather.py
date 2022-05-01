
from detectron2.structures import BoxMode
import os
import natsort
import glob
import json
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import cv2
import random
import csv
import pandas as pd
df = pd.read_csv ('weather.csv')
res=df.query('Weather == "snow"')
ids={"Car":0,"Truck":1,"Bus":2,"Pedestrian":3,"Motorcyclist":4,"Cyclist":5}
def get_amodal_dicts(img_dir):
    dataset_dicts = []
    #files=os.listdir("/home/cad297/Skynet_Data_Decoder/amodal_Ithaca365/")
    with open("/home/cad297/Skynet_Data_Decoder/split/"+img_dir+".txt") as csv_file:
        csv_reader = csv.reader(csv_file)
        rows = list(csv_reader)
        #print(rows)
        for file in res["Date"]:
            #path1=natsort.natsorted(glob.glob("/home/cad297/Skynet_Data_Decoder/amodal_Ithaca365/"+file+"/images/*.png"),reverse=False)
            for i in range(len(rows)):
                img_name="/home/cad297/Skynet_Data_Decoder/amodal_Ithaca365/"+file+"/images/"+rows[i][0]+".png"
                json_file = img_name.replace("images","annotations").replace("png","txt")
                record = {}

                filename = img_name
                height, width = cv2.imread(filename).shape[:2]

                record["file_name"] = filename
                record["image_id"] = filename
                record["height"] = height
                record["width"] = width
                objs = []
                with open(json_file) as f:
                    imgs_anns = json.load(f)
                    for k in range(len(imgs_anns)):
                        poly=imgs_anns[k]['data']
                        '''
                        minx=min(poly, key=lambda x: (x[0], -x[1]))[0]
                        maxx=max(poly, key=lambda x: (x[0], -x[1]))[0]
                        maxy=max(poly, key=lambda y: (y[0], -y[1]))[1]
                        miny=min(poly, key=lambda y: (y[0], -y[1]))[1]
                        '''
                        minx=min(poly, key=lambda x: (x[0], -x[0]))[0]
                        maxx=max(poly, key=lambda x: (x[0], -x[0]))[0]
                        maxy=max(poly, key=lambda y: (y[1], -y[1]))[1]
                        miny=min(poly, key=lambda y: (y[1], -y[1]))[1]
                        poly = [p for x in poly for p in x]
                        obj = {
                            "bbox": [minx,miny,maxx,maxy],
                            "bbox_mode": BoxMode.XYXY_ABS,
                            "segmentation": [poly],
                            "category_id": ids[imgs_anns[k]['class']],
                        }
                        objs.append(obj)
                record["annotations"] = objs
                dataset_dicts.append(record)
    return dataset_dicts

for d in ["train","val"]:
    DatasetCatalog.register("amodal_" + d, lambda d=d: get_amodal_dicts(d))
    MetadataCatalog.get("amodal_" + d).set(thing_classes=["Car","Truck","Bus","Pedestrian","Motorcyclist","Cyclist"])
balloon_metadata = MetadataCatalog.get("amodal_train")
dataset_dicts = get_amodal_dicts("train")
'''
for d in random.sample(dataset_dicts, 1):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2.imwrite("test.png",out.get_image()[:, :, ::-1])
'''
from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("amodal_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 280000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

cfg.OUTPUT_DIR="sunny_train1/"

'''
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
'''
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)
from detectron2.utils.visualizer import ColorMode
dataset_dicts = get_amodal_dicts("val")

for d in dataset_dicts:
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=balloon_metadata,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite("output/sun2snow/"+d["file_name"].split("/")[-3]+"_"+d["file_name"].split("/")[-1],out.get_image()[:, :, ::-1])

'''
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("amodal_val", output_dir="./output_sunny_rain")
val_loader = build_detection_test_loader(cfg, "amodal_val")
print(inference_on_dataset(predictor.model, val_loader, evaluator))
'''
