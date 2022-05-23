from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
import random
import os

from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F
import torch


from ScratchDetector import FasterRCNN


def create_app():
    app = Flask(__name__)
    UPLOAD_FOLDER = 'uploads'
    PROCESSED_FOLDER = 'processed'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    app.category_map = get_scratch_map()
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS
    app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
    app.config['DEVICE'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    app.secret_key = 'super secret key'.encode('utf8')

    # app.model = fasterrcnn_resnet50_fpn(pretrained=True)
    app.model = torch.load('best_iou.pt')
    app.model.eval()
    app.model.to(app.config['DEVICE'])


    @app.route('/')
    def index():
        print('Request for index page received')
        return render_template('index.html')

    @app.route('/favicon.ico')
    def favicon():
        return send_from_directory(os.path.join(app.root_path, 'static'),
                                   'favicon.ico', mimetype='image/vnd.microsoft.icon')

    @app.route('/hello', methods=['POST'])
    def hello():
       name = request.form.get('name')
       surname = request.form.get('surname')

       if name:
           print('Request for hello page received with name=%s' % name)
           return render_template('hello.html', name = name, surname=surname)
       else:
           print('Request for hello page received with no name or blank name -- redirecting')
           return redirect(url_for('index'))

    @app.route('/upload', methods=['GET', 'POST'])
    def upload_file():
        if request.method == 'POST':
            if 'myfile' not in request.files:
                flash('No file part')
                return redirect(url_for('index'))
            file = request.files['myfile']
            # If the user does not select a file, the browser submits an
            # empty file without a filename.
            if file.filename == '':
                flash('No selected file')
                return redirect(url_for('index'))
            if file and allowed_file(file.filename, app.config['ALLOWED_EXTENSIONS']):
                filename = secure_filename(file.filename)
                file.save(os.path.join('static', app.config['UPLOAD_FOLDER'], filename))

                img, boxes, labels = detect_in_image(os.path.join('static', app.config['UPLOAD_FOLDER'], filename))
                bbox_image = show_bboxes(img, boxes, [app.category_map[l] for l in labels])
                bbox_image.save(os.path.join('static', app.config['PROCESSED_FOLDER'], filename))
                flash(f"File {filename} has been processed, detected {' '.join([app.category_map[l] for l in labels])}")
                print(os.path.join('static', app.config['PROCESSED_FOLDER'], filename))
                return render_template('processed_img.html',
                                        image = os.path.join(app.config['PROCESSED_FOLDER'], filename))


    def detect_in_image(img_file, score_threshold=0.5):
        print(img_file)
        img = read_image(img_file)
        img_to_device = img.to(app.config['DEVICE'])
        preds = app.model([img_to_device/255])[0]
        boxes = preds['boxes'][preds['scores'] > score_threshold]
        labels = preds['labels'][preds['scores'] > score_threshold].tolist()
        return img, boxes, labels

    def show_bboxes(img, boxes, labels):
        bbox_img = draw_bounding_boxes(img, boxes, labels, width=5)
        img = bbox_img.detach()
        img = F.to_pil_image(img)
        return img
    return app

def allowed_file(filename, ALLOWED_EXTENSIONS):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_scratch_map():
    return {1:'scratch'}

def get_category_map():
    category_map = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
    }
    return category_map

app = create_app()

if __name__ == '__main__':
    app.run()
