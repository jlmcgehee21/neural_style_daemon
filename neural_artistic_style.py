#!/usr/bin/env python

import os
import json

import config

import argparse
import numpy as np
import scipy.misc
try:
    import deeppy as dp
except:
    pass
import time

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matconvnet import vgg19_net
from style_network import StyleNetwork
import os
import zipfile

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))


def weight_tuple(s):
    try:
        conv_idx, weight = map(float, s.split(','))
        return conv_idx, weight
    except:
        raise argparse.ArgumentTypeError('weights must by "int,float"')


def float_range(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0, 1]" % x)
    return x


def weight_array(weights):
    array = np.zeros(19)
    for idx, weight in weights:
        array[idx] = weight
    norm = np.sum(array)
    if norm > 0:
        array /= norm
    return array


def imread(path):
    return scipy.misc.imread(path).astype(dp.float_)


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, img)


def to_bc01(img):
    return np.transpose(img, (2, 0, 1))[np.newaxis, ...]


def to_rgb(img):
    return np.transpose(img[0], (1, 2, 0))

def process_image_pair(**kwargs):

    start = time.time()

    if kwargs['random_seed'] is not None:
        np.random.seed(kwargs['random_seed'])

    print('loading mat net...')
    layers, img_mean = vgg19_net(kwargs['vgg19'], pool_method='avg')#kwargs['pool_method'])

    # Inputs
    pixel_mean = np.mean(img_mean, axis=(0, 1))
    style_img = imread(kwargs['style_path']) - pixel_mean
    subject_img = imread(kwargs['content_path']) - pixel_mean

    # if kwargs['init'] is None:
    #     init_img = subject_img
    # else:
    #     init_img = imread(kwargs['init']) - pixel_mean

    init_img = subject_img


    noise = np.random.normal(size=init_img.shape, scale=np.std(init_img)*1e-1)
    init_img = init_img * (1 - kwargs['init_noise']) + noise * kwargs['init_noise']

    # Setup network
    print('building network...')

    #FIXME: Set defaults
    subject_weights = weight_array(kwargs['subject_weights']) * kwargs['subject_ratio']
    style_weights = weight_array(kwargs['style_weights'])

    net = StyleNetwork(layers, to_bc01(init_img), to_bc01(subject_img),
                       to_bc01(style_img), subject_weights, style_weights,
                       kwargs['smoothness'])

    # Repaint image
    def net_img():
        return to_rgb(net.image) + pixel_mean

    if kwargs['animation']:
        os.mkdir(kwargs['animation_dir'])

    params = net.params
    learn_rule = dp.Adam(learn_rate=kwargs['learn_rate'])
    learn_rule_states = [learn_rule.init_state(p) for p in params]

    for i in range(kwargs['iterations']):
        print('iteration %d', i)

        if kwargs['animation']:
            imsave(os.path.join(kwargs['animation_dir'], '%.4d.png' % i), net_img())

        cost = np.mean(net.update())
        for param, state in zip(params, learn_rule_states):
            learn_rule.step(param, state)
        print('Iteration: %i, cost: %.4f' % (i, cost))

    imsave(kwargs['output_path'], net_img())

    print("Time Elapsed: %0.2f", time.time()-start)

def make_gif(imgs_dir, save_gif=True, corners=None):

    imgs_dir = '/Users/jeff/Desktop/bell_paintings/results'

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()

    imgs = []
    for file_name in np.asarray(os.listdir(imgs_dir))[::10]:
        if os.path.splitext(file_name)[-1] == '.JPG':

            # Remember that I had to do the color conversion and flip for iPhone image
            img = to_rgb(imread(os.path.join(imgs_dir, file_name)))
            imgs.append(img)

    ims = map(lambda x: (ax.imshow(x), ax.set_title('')), imgs)

    fig.tight_layout()

    im_ani = animation.ArtistAnimation(fig, ims, interval=800, repeat_delay=0, blit=False)


    if save_gif:
        im_ani.save('animation.gif', writer='imagemagick')

    return True





def run(path_to_monitor):
    while True:
        try:


            dirs = map(lambda x: os.path.join(path_to_monitor, x), filter(lambda x: x[0] != '.', os.listdir(path_to_monitor)))

            # Don't process any dirs that are finished
            dirs = filter(lambda dir: '.finished' not in os.listdir(dir), dirs)

            times = []
            for path in dirs:
                times.append(os.path.getctime(path))

            dir_to_process = dirs[np.asarray(times).argsort()[0]]

            style_dir = os.path.join(dir_to_process, 'style')
            content_dir = os.path.join(dir_to_process, 'content')

            try:
                with open(os.path.join(dir_to_process, 'style_options.json'), 'rb') as opt_file:
                    options_dict = json.load(opt_file)
            except:
                options_dict = config.DEFAULTS


            style_paths = map(lambda x: os.path.join(style_dir, x),
                              filter(lambda file: '.jpg' in file and file[0] != '.', os.listdir(style_dir)))

            content_paths = map(lambda x: os.path.join(content_dir, x),
                                filter(lambda file: '.jpg' in file and file[0] != '.', os.listdir(content_dir)))

            print(style_paths)
            print(content_paths)

            if len(style_paths) == 1:
                style_paths = style_paths * len(content_paths)

            if len(content_paths) == 1:
                content_paths = content_paths * len(style_paths)

            print(style_paths)
            print(content_paths)

            #Assert lengths are equal!
            assert len(content_paths) == len(style_paths), "Improper amounts of style/content images"

            results_dir = os.path.join(dir_to_process, "results")

            try:
                os.mkdir(results_dir)
            except:
                pass

            for ind, style_path, content_path in zip(range(len(style_paths)), style_paths, content_paths):
                output_dir = os.path.join(results_dir, 'result_%d' % ind)
                os.mkdir(output_dir)

                options_dict['subject_weights'] = [(9, 1)]

                options_dict['style_weights'] = [(0, 1), (2, 1), (4, 1), (8, 1), (12, 1)]

                options_dict['vgg19'] = os.path.join(FILE_DIR, 'imagenet-vgg-verydeep-19.mat')
                options_dict['style_path'] = style_path
                options_dict['content_path'] = content_path
                options_dict['output_path'] = os.path.join(output_dir,
                                                           os.path.splitext(os.path.basename(style_path))[0] +
                                                           '_final.jpg')

                options_dict['animation_dir'] = os.path.join(output_dir, 'animation')

                process_image_pair(**options_dict)

                if os.path.isdir(options_dict['animation_dir']):
                    # Animation exists
                    pass
                    # make_gif(options_dict['animation_dir'])

            zipf = zipfile.ZipFile(os.path.join(dir_to_process, 'results.zip'), 'w')
            zipdir(results_dir, zipf)
            zipf.close()

            with open(os.path.join(dir_to_process, '.finished'), 'w') as done_file:
                done_file.write('done')


        except Exception as e:
            print(e)
            time.sleep(60)


if __name__ == "__main__":
    # run(config.DIR_TO_PROCESS)

    make_gif('ah')