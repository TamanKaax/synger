
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 14:09:24 2018

@author: Francis Buysse - francis.buysse@gmail.com
"""
import os
from fastai.imports import *
from fastai.conv_learner import *
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import rotate
from skimage.measure import regionprops
from skimage import filters, color, measure, io
from skimage.draw import circle_perimeter
from scipy import ndimage
from flask import url_for
from io import BytesIO
import base64
from app import app


def prepimg(path, frame, col=True, crop=True, rota=False, gaussian=True):
    '''
    Preprocesses image for further use in application

    Args:
        path:     path of one particular image
        frame:    coordinates of a particular frame
        col:      boolean flag for rgb2gray
        crop:     boolean flag for selecting frame instead of complete image
        rota:     boolean flag for adjusting rotation of image
        gaussian: boolean flag for applying gaussian filter

    Returns:
        Preprocessed image
    '''

    img = io.imread(path)
    if rota:
        img = rotate(img, +5)
    if crop:
        img=img[frame[0]:frame[1], frame[2]:frame[3]]
    if col:
        img = img[:,:,0]
        img = color.rgb2gray(img)
    if gaussian:
        img = filters.gaussian(img, 3)
    return img

def extend_slices(slices, extension=60, relative=False):
    '''
    Extends bounding box of a seed object

    Args:
        slices:    list of bounding boxes (slices) of seeds as identified by
                   ndimage.find_objects
        extension: amount of pixels to widen bounding box
        relative:  boolean flag, if False increases bounding box by extension, else
                   increases bounding box relatively to size 96*96

    Returns:
        list of extended bounding boxes (slices)
    '''

    slices_adj = []
    if type(slices) is not list:
        slices = [slices]
    for sli in slices:
        coord = []
        for ax in sli:
            if relative:
                to_add = int((96 - (ax.stop - ax.start))/2)
            else:
                to_add = extension
            coord.append(ax.start-to_add)
            if (relative) and (ax.stop-ax.start+2*to_add < 96):
                coord.append(ax.stop+to_add+1)
            else:
                coord.append(ax.stop+to_add)
        slices_adj.append((slice(coord[0], coord[1]), slice(coord[2], coord[3])))
    return slices_adj

def labeling(img):
    '''
    Otsu thresholding and subsequent labeling of image

    Args:
        img: preprocessed image

    Returns:
        Tuple of label image and quantity of identified labels
    '''

    T = filters.threshold_otsu(img)
    labeled, nr_obj = ndimage.label(img>T)
    return (labeled, nr_obj)

def find_setups(init_img, show=False):
    """
    Finds the 6 frames in an image by using the YUV colorspace

    Args:
        init_img: filename of first image
        show: boolean flag for plotting of identified frames

    Returns:
        List of frame corners coordinates
    """

    setups = []
    img = mpimg.imread(f'{init_img}')
    img = color.rgb2yuv(img)[:, :, 1]
    img = filters.gaussian(img, 10)
    t = filters.threshold_otsu(img)
    img = img > t
    label = measure.label(img)
    regions = measure.regionprops(label)
    avg = []
    # detection might add small non-panel areas and these must be removed
    for r in regions:
        avg.append(r.area)
    avg = np.array(avg).mean()
    for r in regions:
        if r.area > 0.9*avg:
            setups.append([r.bbox[0]+10, r.bbox[2]-10, r.bbox[1]+10, r.bbox[3]-10])
            if show:
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(img[r.bbox[0]+10:r.bbox[2]-10, r.bbox[1]+10:r.bbox[3]-10])
                ax.grid(False)
                plt.show()
    return setups

def initialize_set(init_img, frame):
    '''
    Builds tracking library for all seeds

    Args:
        path:  path to directory of imagestack
        frame: list of coordinates of one frame

    Returns:
        Library of seeds with for each seed the (extended) bounding box, the
        orientation, the centroid, the status and the label
    '''

    init_img = prepimg(f'{init_img}', frame, col=True, rota=False, gaussian=True)
    init_labels, init_nr = labeling(init_img)
    init_slice = ndimage.find_objects(init_labels)
    init_slice_copy = []
    #remove small artifacts
    for s in init_slice:
        if (s[0].stop-s[0].start > 5) and (s[1].stop-s[1].start > 5):
            init_slice_copy.append(s)

    init_extended_slices = extend_slices(init_slice_copy, 30, True)
    init_slices = extend_slices(init_slice_copy, 10)
    tracking_library = {}
    for r in range(len(init_slices)):
        try:
            tracking_library.setdefault(r, {})
            seed_regions = regionprops(init_labels[init_extended_slices[r]])
            tracking_library[r]['ext_slice'] = init_extended_slices[r]
        except:
            seed_regions = regionprops(init_labels[init_slices[r]])
            tracking_library[r]['ext_slice'] = init_slices[r]
        orient = seed_regions[0].orientation*-57.3
        tracking_library[r]['orient'] = orient
        tracking_library[r]['centroid'] = (int(seed_regions[0].centroid[0]), int(seed_regions[0].centroid[1]))
        tracking_library[r]['status'] = 0
        tracking_library[r]['label'] = None
    return tracking_library

def draw_circle(img_out, coord):
    midl, midw = ((coord[0].stop-coord[0].start)/2, (coord[1].stop-coord[1].start)/2)
    rr, cc = circle_perimeter(int(coord[0].start+midl), int(coord[1].start+midw), 10)
    rr2, cc2 = circle_perimeter(int(coord[0].start+midl), int(coord[1].start+midw), 15)
    img_out[rr, cc, 1] = 255
    img_out[rr2, cc2, 1] = 255

def pad(seed):
    return np.pad(seed, ((96-seed.shape[0], 0), (96-seed.shape[1], 0), (0, 0)), mode='constant')

def get_data(sz, test_name, dir_path):
    PATH = dir_path
    arch = resnet50
    tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on)
    return ImageClassifierData.from_paths(PATH, tfms=tfms_from_model(arch, sz), test_name=test_name, trn_name='train')

def load_model():
    arch = resnet34
    data = get_data(96, 'test', f'{app.static_folder}/modeldata')
    learner = ConvLearner.pretrained(arch, data)
    learner.load(f'{app.static_folder}/germix-germcopies_resnet34')
    return learner

def predict(learner, seeds, seeds_indexes):
    '''
    Predict classification of seeds. Seeds_indexes is needed to make match with
    tracking_library as learner.predict uses different order.

    Args:
        learner: pretrained fast.ai learner object
        seeds: list of seeds to be classified as numpy arrays
        seeds_indexes: indexes of seeds as used in tracking_library

    Returns:
        Tuple of predictions and indexes
    '''

    PATH = f'{app.static_folder}/modeldata'
    files = glob.glob(f'{PATH}/test/*')
    for f in files:
        os.remove(f)
    for s, i in zip(seeds, seeds_indexes):
        io.imsave(f'{PATH}/test/{i}.jpg', s)
    learner.set_data(get_data(96, 'test', f'{app.static_folder}/modeldata'))
    data = learner.data
    seeds_indexes = [int(f[-6:-4].strip('/')) for f in data.test_ds.fnames]
    return np.argmax(learner.predict(is_test=True), axis=1), seeds_indexes

def logic(dir_path, interval=1, begin_frame=0, end_frame=6, show=True):
    imgs = sorted(glob.glob(f'{dir_path}/*.jpg'))
    init_img = imgs[0]
    learner = load_model()
    setups = find_setups(init_img)
    results = {}
    for f in range(begin_frame, end_frame):
        frame = setups[f]
        results[f] = {}
        tracking_library = initialize_set(init_img, frame)
        for im in imgs[::interval][:4]:
            germinated = 0
            img = prepimg(f'{im}', frame, col=False, rota=False, gaussian=False)
            img_out = img.copy()
            seeds = []
            seeds_indexes = []
            for s in tracking_library:
                if tracking_library[s]['status'] == 0:
                    seed = img[tracking_library[s]['ext_slice']]
                    if seed.shape != (96, 96, 3):
                        seed = pad(seed)
                    seeds.append(seed)
                    seeds_indexes.append(s)
                else:
                    if tracking_library[s]['status'] == 1:
                        if show:
                            draw_circle(img_out, tracking_library[s]['ext_slice'])
                        germinated += 1
            if seeds:
                pred, seeds_indexes = predict(learner, seeds, seeds_indexes)
                for pr, s2 in zip(pred, seeds_indexes):
                    if pr == 0:
                        if show:
                            draw_circle(img_out, tracking_library[s2]['ext_slice'])
                        tracking_library[s2]['status'] = 1
                        germinated += 1
            results[begin_frame][int(im[-8:-4])] = germinated
            if show:
                fig, ax = plt.subplots(figsize=(8,8))
                ax.imshow(img_out)
                plt.show()
    return results, setups

def logic_setups_only(dir_path):
    imgs = sorted(glob.glob(f'{dir_path}/*.jpg'))
    init_img = imgs[0]
    setups = find_setups(init_img)
    return setups

if __name__ == '__main__':
    logic('/home/taman/data/Germix/Tomato/tom2/', interval=1, f=6, show=True)
