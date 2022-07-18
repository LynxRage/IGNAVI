from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import sys
import time
import argparse
import numpy as np

import cv2
from scipy import ndimage
from skimage.transform import resize
import matplotlib.pyplot as plt

plasma = plt.get_cmap('plasma')
greys = plt.get_cmap('Greys')

# UI and OpenGL
from PySide2 import QtCore, QtGui, QtWidgets, QtOpenGL
from OpenGL import GL, GLU
from OpenGL.arrays import vbo
from OpenGL.GL import shaders
import glm

from utils import post_process_depth, flip_lr
from networks.NewCRFDepth import NewCRFDepth


# Argument Parser
parser = argparse.ArgumentParser(description='NeWCRFs Live 3D')
parser.add_argument('--model_name',      type=str,   help='model name', default='newcrfs')
parser.add_argument('--encoder',         type=str,   help='type of encoder, base07, large07', default='large07')
parser.add_argument('--max_depth',       type=float, help='maximum depth in estimation', default=10)
parser.add_argument('--checkpoint_path', type=str,   help='path to a checkpoint to load', required=True)
parser.add_argument('--input_height',    type=int,   help='input height', default=480)
parser.add_argument('--input_width',     type=int,   help='input width',  default=640)
parser.add_argument('--dataset',         type=str,   help='dataset this model trained on',  default='nyu')
parser.add_argument('--crop',            type=str,   help='crop: kbcrop, edge, non',  default='non')
parser.add_argument('--video',           type=str,   help='video path',  default='')

args = parser.parse_args()

# Image shapes
height_rgb, width_rgb = args.input_height, args.input_width
height_depth, width_depth = height_rgb, width_rgb

def load_model():
    args.mode = 'test'
    model = NewCRFDepth(version='large07', inv_depth=False, max_depth=args.max_depth)
    model = torch.nn.DataParallel(model)

    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()
    
    return model

# Function timing
ticTime = time.time()


def tic():
    global ticTime;
    ticTime = time.time()


def toc():
    print('{0} seconds.'.format(time.time() - ticTime))


# Conversion from Numpy to QImage and back
def np_to_qimage(a):
    im = a.copy()
    return QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888).copy()


def qimage_to_np(img):
    img = img.convertToFormat(QtGui.QImage.Format.Format_ARGB32)
    return np.array(img.constBits()).reshape(img.height(), img.width(), 4)


# Compute edge magnitudes
def edges(d):
    dx = ndimage.sobel(d, 0)  # horizontal derivative
    dy = ndimage.sobel(d, 1)  # vertical derivative
    return np.abs(dx) + np.abs(dy)


# Main window
class Window(QtWidgets.QWidget):
    updateInput = QtCore.Signal()
    
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.model = None
        self.capture = None
        
        mainLayout = QtWidgets.QVBoxLayout()
        
        # Input / output views
        viewsLayout = QtWidgets.QGridLayout()
        self.inputViewer = QtWidgets.QLabel("[Click to start]")
        self.inputViewer.setPixmap(QtGui.QPixmap(width_rgb, height_rgb))
        self.outputViewer = QtWidgets.QLabel("[Click to start]")
        self.outputViewer.setPixmap(QtGui.QPixmap(width_rgb, height_rgb))
        
        imgsFrame = QtWidgets.QFrame()
        inputsLayout = QtWidgets.QVBoxLayout()
        imgsFrame.setLayout(inputsLayout)
        inputsLayout.addWidget(self.inputViewer)
        inputsLayout.addWidget(self.outputViewer)
        
        viewsLayout.addWidget(imgsFrame, 0, 0)
        #viewsLayout.setColumnStretch(1, 10)
        mainLayout.addLayout(viewsLayout)
        
        # Load depth estimation model
        toolsLayout = QtWidgets.QHBoxLayout()

        self.button3 = QtWidgets.QPushButton("Video")
        self.button3.clicked.connect(self.loadVideoFile)
        toolsLayout.addWidget(self.button3)
        
        self.button4 = QtWidgets.QPushButton("Pause")
        self.button4.clicked.connect(self.loadImage)
        toolsLayout.addWidget(self.button4)
        
        self.button6 = QtWidgets.QPushButton("Refresh")
        self.button6.clicked.connect(self.updateCloud)
        toolsLayout.addWidget(self.button6)
        
        mainLayout.addLayout(toolsLayout)
        
        self.setLayout(mainLayout)
        self.setWindowTitle(self.tr("IGNAVI with NeWCRFs"))
        
        # Signals
        self.updateInput.connect(self.update_input)
    
    def loadModel(self):
        print('== loadModel')
        QtGui.QGuiApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        tic()
        self.model = load_model()
        print('Model loaded.')
        toc()
        self.updateCloud()
        QtGui.QGuiApplication.restoreOverrideCursor()
    
    def loadImage(self):
        print('== loadImage')
        self.capture = None
        self.inputViewer.setPixmap(QtGui.QPixmap.fromImage(np_to_qimage(img)))
        self.updateCloud()
    
    def loadVideoFile(self):
        print('== loadVideoFile')
        self.model = load_model()
        self.capture = cv2.VideoCapture(args.video)
        self.updateInput.emit()
    
    def update_input(self):
        print('== update_input')
        # Don't update anymore if no capture device is set
        if self.capture == None:
            return
        
        # Capture a frame
        ret, frame = self.capture.read()
        # Loop video playback if current stream is video file
        if not ret:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.capture.read()
        
        frame_ud = cv2.resize(frame, (width_rgb, height_rgb), interpolation=cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame_ud, cv2.COLOR_BGR2RGB)
        image = np_to_qimage(frame)
        self.inputViewer.setPixmap(QtGui.QPixmap.fromImage(image))
        
        # Update the point cloud
        self.updateCloud()
    
    def updateCloud(self):
        print('== updateCloud')
        rgb8 = qimage_to_np(self.inputViewer.pixmap().toImage())
        
        if self.model:
            input_image = rgb8[:, :, :3].astype(np.float32)

            # Normalize image
            input_image[:, :, 0] = (input_image[:, :, 0] - 123.68) * 0.017
            input_image[:, :, 1] = (input_image[:, :, 1] - 116.78) * 0.017
            input_image[:, :, 2] = (input_image[:, :, 2] - 103.94) * 0.017

            H, W, _ = input_image.shape
            if args.crop == 'kbcrop':
                top_margin = int(H - 352)
                left_margin = int((W - 1216) / 2)
                input_image_cropped = input_image[top_margin:top_margin + 352, 
                                                  left_margin:left_margin + 1216]
            elif args.crop == 'edge':
                input_image_cropped = input_image[32:-32, 32:-32, :]
            else:
                input_image_cropped = input_image

            input_images = np.expand_dims(input_image_cropped, axis=0)
            input_images = np.transpose(input_images, (0, 3, 1, 2))

            with torch.no_grad():
                image = Variable(torch.from_numpy(input_images)).cuda()
                # Predict
                depth_est = self.model(image)
                post_process = True
                if post_process:
                    image_flipped = flip_lr(image)
                    depth_est_flipped = self.model(image_flipped)
                    depth_cropped = post_process_depth(depth_est, depth_est_flipped)

            depth = np.zeros((height_depth, width_depth), dtype=np.float32)
            if args.crop == 'kbcrop':
                depth[top_margin:top_margin + 352, left_margin:left_margin + 1216] = \
                        depth_cropped[0].cpu().squeeze() / args.max_depth
            elif args.crop == 'edge':
                depth[32:-32, 32:-32] = depth_cropped[0].cpu().squeeze() / args.max_depth
            else:
                depth[:, :] = depth_cropped[0].cpu().squeeze() / args.max_depth

            coloredDepth = (greys(np.log10(depth * args.max_depth))[:, :, :3] * 255).astype('uint8')
            self.outputViewer.setPixmap(QtGui.QPixmap.fromImage(np_to_qimage(coloredDepth)))
            cv2.imwrite("depth_out.png", coloredDepth)
        
        # Update to next frame if we are live
        QtCore.QTimer.singleShot(10, self.updateInput)
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    res = app.exec_() 
