import cv2
from PIL import Image
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe,Value,Array
import torch
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank
import os
from read_input import get_inputs
import numpy as np

#TODO: change splits to dirname

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-s", "--save", help="whether save",action="store_true")
    parser.add_argument('-th','--threshold',help='threshold to decide identical faces',default=1.54, type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank",action="store_true")
    parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true")
    parser.add_argument("-c", "--score", help="whether show the confidence score",action="store_true")
    args = parser.parse_args()

    conf = get_config(False)

    mtcnn = MTCNN()
    print('mtcnn loaded')
    
    learner = face_learner(conf, True)
    learner.threshold = args.threshold
    if conf.device.type == 'cpu':
        learner.load_state(conf, 'cpu_final.pth', True, True)
    else:
        learner.load_state(conf, 'final.pth', True, True)
    learner.model.eval()
    print('learner loaded')

    # todo, modify this
    filepath = './input.txt'
    # read files
    train_dict, test_dict, output_dir = get_inputs(filepath)

    if args.update:
        targets, names = prepare_facebank(conf, learner.model, mtcnn, tta = args.tta,
                                          load_from_custom_dir = True, dir_name_dict = train_dict)
        print('facebank updated')
    else:
        targets, names = load_facebank(conf)
        print('facebank loaded')

    id_results = []
    random_output_idx = 0
    for tc_id in test_dict:
        tc_path = test_dict[tc_id]
        result = str(tc_id)
        image = cv2.imread(tc_path, cv2.IMREAD_COLOR)  # queryImage
        try:
#           image = Image.fromarray(frame[...,::-1]) #bgr to rgb
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            bboxes, faces = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)
            bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibility faces
            bboxes = bboxes.astype(int)
            bboxes = bboxes + [-1,-1,1,1] # personal choice
            results, score = learner.infer(conf, faces, targets, args.tta, k=conf.num_of_guesses)
            for idx,bbox in enumerate(bboxes):
                for name in names[results[idx] + 1]:
                    result += ' ' + str(name)
        except Exception as e:
            result += ' ' + conf.nfd
            for i in range(conf.num_of_guesses - 1):
                result += ' ' + str(names[(random_output_idx + i) % len(names)])
            random_output_idx += conf.num_of_guesses - 1
        id_results.append(result)
    print(output_dir)
    with open(output_dir, 'a') as output_file:
        for result in id_results:
            # todo: check if the last \n is a problem
            output_file.write(result + '\n')
