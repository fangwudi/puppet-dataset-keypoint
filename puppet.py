#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__title__ = "puppet"
__author__ = 'fangwudi'
__time__ = '17-12-5 16：20'

       code is far away from bugs
             ┏┓   ┏┓
            ┏┛┻━━━┛┻━┓
            ┃   ━    ┃
            ┃ ┳┛  ┗┳ ┃
            ┃    ┻   ┃
            ┗━┓    ┏━┛
              ┃    ┗━━━━━┓
              ┃          ┣┓
              ┃          ┏┛
              ┗┓┓┏━━┳┓┏━━┛
               ┃┫┫  ┃┫┫
               ┗┻┛  ┗┻┛
     with the god animal protecting
     
"""
import random
import math
import threading
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt


class PuppetDataset(object):
    """Generates puppet dataset. The dataset consists of random number of puppet
    placed randomly on a blank surface. The images are generated on the fly.
    No file access required. Puppet here is just a name, and actually you can
    diy your own instance having keypoint.
    1 puppet has 4 keypoints(hand, shoulder, waist, foot),
             and 4 parts(hand, arm, body, leg)
    use next() function to generate a batch
    return argument see produce_one() function, meaning:
    (1) image: image for train
    (2) mask: mask array including each puppet in 1 image
    (3) annkp: tupple, first as keypoint coord, second as keypoint visible state
    """
    def __init__(self, batch_size, height, width, order=True, seed=None):
        """Generate the requested number of synthetic images.
        batch_size: number of images to generate.
        height, width: the size of the generated images.
        order: use size of object to decide draw it near or faraway
        """
        if seed is None:
            seed = np.uint32((time.time() % 1)) * 1000
        np.random.seed(seed)
        self.batch_size = batch_size
        self.height = height
        self.width = width
        # next threading
        self.batch_index = 0
        self.lock = threading.Lock()
        # puppet begin
        self.order = order
        # (1)decide min max control size of object
        puppetsize = 23  # just an estimate
        self.minsize = 1
        self.maxsize = max(self.minsize, min(height, width)//puppetsize)
        # (2)decide min max number of object
        self.minnum_ob = 2
        self.maxnum_ob = 5

    def produce_one(self):
        plans = self.plan_canvas()
        n = len(plans)  # n puppet
        # keypoint annotation
        kp = [plan[1][:4] for plan in plans]
        kpv = np.zeros([n, 4], dtype=np.uint8)  # visible
        # generate bg
        bg_color = np.array([random.randint(0, 255)
                             for _ in range(3)]).reshape([1, 1, 3])
        image = np.ones([self.height, self.width, 3], dtype=np.uint8)
        image = image * bg_color.astype(np.uint8)
        # init masks
        mask = np.zeros([self.height, self.width, n], dtype=np.uint8)
        # draw puppet
        for i, plan in enumerate(plans):
            image = self.draw_puppet(image, plan)
            mask[:, :, i] = self.draw_puppet(mask[:, :, i].copy(),
                                             plan, mask_flag=True)
        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(n-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            for j in range(4):
                x = kp[i][j][0]
                y = kp[i][j][1]
                # if outside bound set 0 same as occlusion
                if x < 0 or y < 0 or x >= self.width or y >= self.height:
                    kpv[i][j] = 0
                else:
                    kpv[i][j] = mask[x, y, i]
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        # merge kp annotation
        annkp = (kp, kpv)
        return image, mask, annkp

    @staticmethod
    def draw_puppet(image, plan, mask_flag=False):
        """Draws a puppet."""
        s, coords, colors, other = plan
        if mask_flag:
            colors = [1 for _ in range(4)]
        # draw arm and leg
        cv2.line(image, tuple(coords[0]), tuple(coords[1]), colors[1], s)
        cv2.line(image, tuple(coords[2]), tuple(coords[3]), colors[3], s)
        # draw hand
        x = coords[0][0]
        y = coords[0][1]
        points = np.array([[(x, y-s),
                            (x - 2*s, y + s),
                            (x + 2*s, y + s),
                            ]], dtype=np.int32)
        cv2.fillPoly(image, points, colors[0])
        # draw body
        cv2.ellipse(image, tuple(coords[4]), other['ellipse_axis'],
                    other['ellipse_angle'], 0, 360, colors[2], -1)
        return image

    def plan_canvas(self):
        # generate number of object,
        num_ob = random.randint(self.minnum_ob, self.maxnum_ob)
        # generate size of object
        sizes = [random.randint(self.minsize, self.maxsize)
                 for _ in range(num_ob)]
        # place bigger object near ; draw from faraway to near
        if self.order:
            sizes.sort()
        plans = [self.plan_puppet(size) for size in sizes]
        return plans

    def plan_puppet(self, size):
        # init part lenth
        pl1 = 8  # arm
        pl2 = 9  # body
        pl3 = 7  # leg
        # decide joint angles
        range_ja1 = list(range(-45, 45)) + list(range(135, 225))
        range_ja2 = list(range(45, 135))
        range_ja3 = list(range(60, 120))
        ja1 = random.choice(range_ja1)
        ja2 = random.choice(range_ja2)
        ja3 = random.choice(range_ja3)
        # decide part color
        # 0:hand, 1:arm, 2:body 3:leg
        colors = [tuple([random.randint(0, 255) for _ in range(3)])
                  for _ in range(4)]

        def nextcoord(here, lenth, angle):
            x = here[0]+lenth*math.cos(math.radians(angle))
            y = here[1]+lenth*math.sin(math.radians(angle))
            return round(x), round(y)
        # count keypoint coord
        c0 = (0, 0)  # hand
        c1 = nextcoord(c0, pl1*size, ja1)  # shoulder
        c2 = nextcoord(c1, pl2*size, ja2)  # waist
        c3 = nextcoord(c2, pl3*size, ja3)  # foot
        ec = nextcoord(c1, pl2*size/3, ja2)
        coords = np.array([c0, c1, c2, c3, ec])
        other = {'ellipse_angle': ja2-180,
                 'ellipse_axis': (int(pl2*size*2/3), int(pl2*size/3))}
        # decide at least which keypoint to show and its coord
        i = random.randint(0, 3)
        # Center x, y
        buffer = 2  # margin
        cy = random.randint(buffer, self.height - buffer - 1)
        cx = random.randint(buffer, self.width - buffer - 1)
        center = np.array([cx, cy])
        bias = center - coords[i]
        coords[:, ] += bias
        return size, coords, colors, other

    def __next__(self):
        return self.next()

    def next(self):
        with self.lock:
            self.batch_index += 1
        image_group = []
        mask_group = []
        annkp_group = []
        for _ in range(self.batch_size):
            out = self.produce_one()
            image_group.append(out[0])
            mask_group.append(out[1])
            annkp_group.append(out[2])
        return image_group, mask_group, annkp_group


def display_images(images, cols=5, cmap=None, norm=None,
                   interpolation=None):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    titles: optional. A list of titles to display with each image.
    cols: number of images per row
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interporlation to use for display.
    """
    plt.figure(figsize=(14, 14 * cols))
    i = 1
    for image in images:
        plt.subplot(1, cols, i)
        plt.axis('off')
        plt.imshow(image.astype(np.uint8), cmap=cmap,
                   norm=norm, interpolation=interpolation)
        i += 1
    plt.show()


def display_top_masks(image, mask, limit=5):
    """Display the given image and the top few class masks."""
    to_display = [image]
    n = len(mask[0][0])
    # Generate images and titles
    for i in range(n-1, max(n-limit-1, -1), -1):
        m = mask[:, :, i]
        to_display.append(m)
    display_images(to_display, cols=limit + 1, cmap="Blues_r")


def main():
    # build dataset
    batch = 6
    height = 48
    width = 48
    dataset = PuppetDataset(batch, height, width)
    # generate and display
    image_group, mask_group, _ = dataset.next()
    for x in range(batch):
        image = image_group[x]
        mask = mask_group[x]
        display_top_masks(image, mask)
