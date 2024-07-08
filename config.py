import os
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import tensorflow as tf

class Config:
    SECRET_KEY = 'hehehehe'
    UPLOAD_FOLDER = 'static/uploads'
    ALLOWED_EXTENSIONS = {'mp4'}

    SQL_SERVER = 'YOUR_SQL_SERVER'
    SQL_USER = 'YOUR_SQL_USER'
    SQL_PASSWORD = 'YOUR_SQL_PASSWORD'
    SQL_DATABASE = 'bibdata'

    BIB_DETECTION_MODEL_PATH = 'models\\bib_detector.pt'  #path to pretrained bib detector
    ESRGAN_MODEL_PATH = "models\\esrgan-tf2\\1" # path to esrgan model
    COCO_MODEL_PATH = "models\\yolov8n.pt" # path to yolov8n model

    # Initialize VietOCR with default weights
    vietocr_config = Cfg.load_config_from_name('vgg_transformer')
    vietocr_config['device'] = 'cuda' if tf.config.list_physical_devices('GPU') else 'cpu'
    vietocr_config['cnn']['pretrained'] = False
    vietocr_config['predictor']['beamsearch'] = False
    detector = Predictor(vietocr_config)