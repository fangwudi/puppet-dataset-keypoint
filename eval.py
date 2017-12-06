#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__title__ = "eval"
__author__ = 'fangwudi'
__time__ = '17-12-5 17：42'

code is far away from bugs 
     ┏┓   ┏┓
    ┏┛┻━━━┛┻━┓
    ┃        ┃
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
import numpy as np
import time


class Eval(object):
    def __init__(self, gtAll=None, dtAll=None):
        """
        Initialize Eval for keypoint
        :param gtAll: object with ground truth annotations
        :param dtAll: object with detection results
        :return: None
        # gt
        [[{"keypoints": [x1,y1,v1,...,xk,yk,vk], "size": int}, ...]...]
        # dt
        [[{"keypoints": [x1,y1,v1,...,xk,yk,vk], "score": float}, ...]...]

        here: v=1 visible and labeled
              v=0 not visible or not labeled
        """
        # delete all 0 gt
        for gts in gtAll:
            for i in range(len(gts)-1, -1, -1):
                vg = gts[i]['keypoints'][2::3]
                if not any(vg):
                    gts.pop(i)
        self.gtAll   = gtAll
        self.dtAll   = dtAll
        self.num_image = len(self.gtAll)
        self.maxDet = 20
        # iou threshold
        self.iouThrs = np.linspace(.5, 0.95, 10)

    def evaluate(self):
        """
        Run all image evaluation on given images
        :return:
        """
        tic = time.time()
        print('Running per image keypoint evaluation...')
        sum_match = np.zeros(len(self.iouThrs), dtype=int)
        sum_object = 0
        for x in range(self.num_image):
            num_match, num_object = self.evaluateImage(self.gtAll[x], self.dtAll[x])
            sum_match += num_match
            sum_object += num_object
        if sum_object <= 0:
            raise Exception("ground truth object 0")
        result = sum_match / float(sum_object)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))
        print("oks threshold: {}".format(self.iouThrs))
        print("corresponding AP: {}".format(result))
        mAP = result.mean()
        print("mean AP: {}".format(mAP))

    def evaluateImage(self, gts, dts):
        """
        perform evaluation for single image
        :param gts: single image ground truth
        :param dts: single image detection
        :return: num_match array, num_object
        """
        # dimention here should be Nxm
        T = len(self.iouThrs)
        num_match = np.zeros(T, dtype=int)
        G = len(gts)
        D = len(dts)
        if G == 0:
            # no groud truth object
            return num_match, 0
        if D == 0:
            # no detection object but have groud truth
            return num_match, D
        # sort detection according to object prob
        inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > self.maxDet:
            dts = dts[0:self.maxDet]
        # init iou
        ious = np.zeros((len(dts), len(gts)))
        # prepare iou fomula
        sigmas = np.array([0.05, 0.05, 0.05, 0.05])  # set keypoint sigma
        puppetarea = 23 * 23
        deno = 2 * puppetarea * (sigmas * 2) ** 2  # iou denominator
        # compute iou between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = gt['keypoints']
            # x1,y1,v1,x2,y2,v2...
            xg = g[0::3]
            yg = g[1::3]
            vg = g[2::3]
            for i, dt in enumerate(dts):
                d = np.array(dt['keypoints'])
                xd = d[0::3]
                yd = d[1::3]
                # measure the per-keypoint distance if keypoints visible
                dx = xd - xg
                dy = yd - yg
                e = (dx ** 2 + dy ** 2) / deno / gt['size']  # need size in
                e = e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        # init match array
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        # find match
        for tind, t in enumerate(self.iouThrs):
            for dind in range(D):
                # information about best match so far (m=-1 -> unmatched)
                temp_iou = t
                m = -1
                for gind in range(G):
                    # if this gt already matched, continue
                    if gtm[tind, gind] > 0:
                        continue
                    # continue to next gt unless better match made
                    if ious[dind, gind] < temp_iou:
                        continue
                    # match successful and best so far, store appropriately
                    temp_iou = ious[dind, gind]
                    m = gind
                # if match made, store id of match for both dt and gt
                if m == -1:
                    continue
                dtm[tind, dind] = m
                gtm[tind, m] = dind
        # count
        num_match = np.count_nonzero(dtm, axis=1)
        return num_match, D
