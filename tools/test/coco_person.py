# Part of this code is derived/taken from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
import os
import pickle
import random
from collections import OrderedDict
from collections import defaultdict

import cv2
import json_tricks as json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision import transforms
from tqdm import tqdm


class COCODataset():
    """
    COCODataset class.
    """

    def __init__(self, 
                 root_path="/database/coco/", data_version="val2017", is_train=True, use_gt_bboxes=True, bbox_path="",
                 image_width=288, image_height=384, color_rgb=True,
                 scale=True, scale_factor=0.35, flip_prob=0.5, rotate_prob=0.5, rotation_factor=45., half_body_prob=0.3,
                 use_different_joints_weight=False, heatmap_sigma=3, soft_nms=False,
                 ):
        self.root_path = root_path
        self.data_version = data_version
        self.is_train = is_train
        self.use_gt_bboxes = use_gt_bboxes
        self.bbox_path = bbox_path
        self.image_width = image_width
        self.image_height = image_height
        self.color_rgb = color_rgb
        self.scale = scale  # ToDo Check
        self.scale_factor = scale_factor
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.rotation_factor = rotation_factor
        self.half_body_prob = half_body_prob
        self.use_different_joints_weight = use_different_joints_weight  # ToDo Check
        self.heatmap_sigma = heatmap_sigma
        self.soft_nms = soft_nms

        self.data_path = os.path.join(self.root_path, self.data_version)
        self.annotation_path = os.path.join(
            self.root_path, 'annotations', 'person_keypoints_%s.json' % self.data_version
        )

        self.image_size = (self.image_width, self.image_height)
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.heatmap_size = (int(self.image_width / 4), int(self.image_height / 4))
        self.heatmap_type = 'gaussian'
        self.pixel_std = 200  # I don't understand the meaning of pixel_std (=200) in the original implementation

        self.nof_joints = 17
        self.nof_joints_half_body = 8
        self.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
        self.upper_body_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.lower_body_ids = [11, 12, 13, 14, 15, 16]
        self.joints_weight = np.asarray(
            [1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5, 1.5],
            dtype=np.float32
        ).reshape((self.nof_joints, 1))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load COCO dataset - Create COCO object then load images and annotations
        self.coco = COCO(self.annotation_path)
        self.imgIds = self.coco.getImgIds()





        self.data_version = "val2017"
        self.data_path = "/database/coco/"
        self.annotation_path = os.path.join(
            self.data_path, 'annotations', 'person_keypoints_%s.json' % self.data_version
        )

        # Load COCO dataset - Create COCO object then load images and annotations
        self.coco = COCO(self.annotation_path)
        self.imgIds = self.coco.getImgIds()

        self.count_people1 =0
        self.count_people2 =0
        self.count_people3 =0
        self.count_people4p =0

        self.data = []
        # load annotations for each image of COCO
        for imgId in tqdm(self.imgIds):

            ann_ids = self.coco.getAnnIds(imgIds=imgId, iscrowd=False)

            img = self.coco.loadImgs(imgId)[0]

            if self.use_gt_bboxes:
                objs = self.coco.loadAnns(ann_ids)

                # sanitize bboxes
                valid_objs = []
                person_object_count =0
                for obj in objs:
                    # Skip non-person objects (it should never happen)
                    if obj['category_id'] != 1:
                        continue

                    # ignore objs without keypoints annotation
                    if max(obj['keypoints']) == 0:
                        continue
                        
                    person_object_count += 1
                    x, y, w, h = obj['bbox']
                    x1 = np.max((0, x))
                    y1 = np.max((0, y))
                    x2 = np.min((img['width'] - 1, x1 + np.max((0, w - 1))))
                    y2 = np.min((img['height'] - 1, y1 + np.max((0, h - 1))))

                    # Use only valid bounding boxes
                    if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                        obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                        valid_objs.append(obj)

                    
                if person_object_count == 1:
                    self.count_people1 +=1
                if person_object_count == 2:
                    self.count_people2 +=1
                if person_object_count == 3:
                    self.count_people3 +=1
                if person_object_count >3:
                    self.count_people4p +=1

                objs = valid_objs

            else:
                objs = bboxes[imgId]

        print('\nCOCO dataset loaded!')
        print("count_people1 : ", self.count_people1)
        print("count_people2 : ", self.count_people2 )
        print("count_people3 : ", self.count_people3)
        print("count_people4 : ", self.count_people4p)

    def evaluate_accuracy(self, output, target, params=None):
        if params is not None:
            hm_type = params['hm_type']
            thr = params['thr']
            accs, avg_acc, cnt, joints_preds, joints_target = evaluate_pck_accuracy(output, target, hm_type, thr)
        else:
            accs, avg_acc, cnt, joints_preds, joints_target = evaluate_pck_accuracy(output, target)

        return accs, avg_acc, cnt, joints_preds, joints_target

    def evaluate_overall_accuracy(self, predictions, bounding_boxes, image_paths, output_dir, rank=0.):

        res_folder = os.path.join(output_dir, 'results')
        if not os.path.exists(res_folder):
            os.makedirs(res_folder, 0o755, exist_ok=True)

        res_file = os.path.join(res_folder, 'keypoints_{}_results_{}.json'.format(self.data_version, rank))

        # person x (keypoints)
        _kpts = []
        for idx, kpt in enumerate(predictions):
            _kpts.append({
                'keypoints': kpt,
                'center': bounding_boxes[idx][0:2],
                'scale': bounding_boxes[idx][2:4],
                'area': bounding_boxes[idx][4],
                'score': bounding_boxes[idx][5],
                'image': int(image_paths[idx][-16:-4])
            })

        # image x person x (keypoints)
        kpts = defaultdict(list)
        for kpt in _kpts:
            kpts[kpt['image']].append(kpt)

        # rescoring and oks nms
        num_joints = self.nof_joints
        in_vis_thre = self.in_vis_thre
        oks_thre = self.oks_thre
        oks_nmsed_kpts = []
        for img in kpts.keys():
            img_kpts = kpts[img]
            for n_p in img_kpts:
                box_score = n_p['score']
                kpt_score = 0
                valid_num = 0
                for n_jt in range(0, num_joints):
                    t_s = n_p['keypoints'][n_jt][2]
                    if t_s > in_vis_thre:
                        kpt_score = kpt_score + t_s
                        valid_num = valid_num + 1
                if valid_num != 0:
                    kpt_score = kpt_score / valid_num
                # rescoring
                n_p['score'] = kpt_score * box_score

            if self.soft_nms:
                keep = soft_oks_nms([img_kpts[i] for i in range(len(img_kpts))], oks_thre)
            else:
                keep = oks_nms([img_kpts[i] for i in range(len(img_kpts))], oks_thre)

            if len(keep) == 0:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

        self._write_coco_keypoint_results(oks_nmsed_kpts, res_file)
        if 'test' not in self.data_version:
            info_str = self._do_python_keypoint_eval(res_file)
            name_value = OrderedDict(info_str)
            return name_value, name_value['AP']
        else:
            return {'Null': 0}, 0

    # Private methods

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2,), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def _half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.nof_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if random.random() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5

        return center, scale

    def _generate_target(self, joints, joints_vis):
        """
        :param joints:  [nof_joints, 2]
        :param joints_vis: [nof_joints, 2]
        :return: target, target_weight(1: visible, 0: invisible)
        """
        target_weight = np.ones((self.nof_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        if self.heatmap_type == 'gaussian':
            target = np.zeros((self.nof_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.heatmap_sigma * 3

            for joint_id in range(self.nof_joints):
                feat_stride = np.asarray(self.image_size) / np.asarray(self.heatmap_size)
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.heatmap_sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        else:
            raise NotImplementedError

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight

    def _write_coco_keypoint_results(self, keypoints, res_file):
        data_pack = [
            {
                'cat_id': 1,  # 1 == 'person'
                'cls': 'person',
                'ann_type': 'keypoints',
                'keypoints': keypoints
            }
        ]

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
        try:
            json.load(open(res_file))
        except Exception:
            content = []
            with open(res_file, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1] = ']'
            with open(res_file, 'w') as f:
                for c in content:
                    f.write(c)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array([img_kpts[k]['keypoints'] for k in range(len(img_kpts))], dtype=np.float32)
            key_points = np.zeros((_key_points.shape[0], self.nof_joints * 3), dtype=np.float32)

            for ipt in range(self.nof_joints):
                key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]
                key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]
                key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]  # keypoints score.

            result = [
                {
                    'image_id': img_kpts[k]['image'],
                    'category_id': cat_id,
                    'keypoints': list(key_points[k]),
                    'score': img_kpts[k]['score'].astype(np.float32),
                    'center': list(img_kpts[k]['center']),
                    'scale': list(img_kpts[k]['scale'])
                }
                for k in range(len(img_kpts))
            ]
            cat_results.extend(result)

        return cat_results

    def _do_python_keypoint_eval(self, res_file):
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[ind]))

        return info_str


if __name__ == '__main__':
    coco = COCODataset(rotate_prob=0., half_body_prob=0.)
