import os
import sys
sys.path.insert(0, os.getcwd())
from script.vqa.create_dic import create_dic
from script.vqa.compute_softscore import compute_softscore
from script.vqa.extract_img_fea import extract_feature
from script.vqa.detection_feature_convert import feature_convert
if __name__ == '__main__':
    extract_feature()
    create_dic()
    compute_softscore()
    feature_convert()