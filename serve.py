from __future__ import print_function
import caffe
import numpy as np
import moxel.space
import json

net = caffe.Net('mobilenet_deploy.prototxt', 'mobilenet.caffemodel', caffe.TEST)
with open('labels.txt', 'r') as f:
    labels = f.readlines()
    labels = [line.replace('\n', '') for line in labels]

def predict(image):
    image.resize((224, 224))
    image = image.to_numpy_rgb()[:, :, :3].transpose(2, 0, 1)
    image = np.array(image, dtype='float32')
    image[:,:,0] -= 103.94
    image[:,:,1] -= 116.78
    image[:,:,2] -= 123.68
    image *= 0.017

    image = np.expand_dims(image, 0)
    net.blobs['data'].data[...] = image
    net.forward()
    pred = net.blobs['fc7'].data
    pred = pred.reshape(1000).tolist()
    indices = np.flip(np.argsort(pred), axis=0)

    top_pred = [(labels[x], float(pred[x])) for x in indices[:5]]

    return {
        'classes': top_pred
    }

