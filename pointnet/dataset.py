import json
import os
import pickle
import sys

import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm


def sample_furthest_points(points: np.ndarray,
                           nsamples: int,
                           random_start_point: bool = True) -> np.ndarray:
    """Furthest point sampling algorithm [1] to iteratively sample nsamples
    points from a given pointcloud.

    Farthest point sampling algorithm provides more uniform coverage of the
    input point cloud compared to uniform random sampling.

    [1] Charles R. Qi et al, "PointNet++: Deep Hierarchical Feature Learning
        on Point Sets in a Metric Space", NeurIPS 2017.

    Args:
        points (np.ndarray): (N, D) array of points
        nsamples (int): number of samples
        random_start_point (bool, optional): to start from a random index.
            Defaults to True.

    Returns:
        np.ndarray: (nsamples, D) array of sampled points
    """
    N = points.shape[0]

    if N < nsamples:
        sampled_indices = np.random.choice(N, nsamples, replace=True)
        sampled_points = points[sampled_indices]
        return sampled_points

    closest_dists = np.full(shape=(N,), fill_value=np.inf, dtype=np.float32)
    selected_idx = np.random.randint(N) if random_start_point else 0
    sampled_indices = np.empty(shape=(nsamples,), dtype=np.int64)
    sampled_indices[0] = selected_idx
    for i in range(1, nsamples):
        # Find the distance between the last selected point and all other points.
        dist_to_last_selected = points - points[selected_idx]
        dist_to_last_selected = np.sum(dist_to_last_selected**2, -1)
        # If closer than currently saved distance to one of the selected
        # points, then updated closest_dists
        dist_mask = dist_to_last_selected < closest_dists
        closest_dists[dist_mask] = dist_to_last_selected[dist_mask]
        # The aim is to pick the point that has the largest nearest neighbour
        # distance to any of the already selected points
        selected_idx = np.argmax(closest_dists)
        sampled_indices[i] = selected_idx

    # Gather the points
    sampled_points = points[sampled_indices]
    return sampled_points


def normalize_points(points: np.ndarray) -> np.ndarray:
    """Enclose all points within a unit sphere.

    Args:
        points (np.ndarray): (N, D) array of points

    Returns:
        np.ndarray: (N, D) array of points
    """
    centroid = np.mean(points, axis=0)
    points -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(points**2, axis=1)))
    points /= furthest_distance  # scale
    return points


def gen_phase_id(root):
    classes = []

    file_name = os.path.join(root, 'train.txt')
    with open(file_name, 'r') as f:
        for line in f:
            classes.append(line.strip().split('/')[0])

    classes = np.unique(classes)

    file_name = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), '..', 'misc', 'phase_id.txt')
    with open(file_name, 'w') as f:
        for i in range(len(classes)):
            f.write('{}\t{}\n'.format(classes[i], i))


class PhaseDataset(data.Dataset):
    def __init__(self,
                 root: str,
                 npoints: int = 1024,
                 classification: bool = False,
                 class_choice: list = None,
                 split: str = 'train',
                 data_augmentation: bool = True,
                 preprocess_data: bool = True,
                 random_sample: bool = True):
        self.npoints: int = npoints
        self.classification: bool = classification
        self.data_augmentation: bool = data_augmentation
        self.preprocess_data: bool = preprocess_data
        self.random_sample: bool = random_sample

        phase_category_file: str = os.path.join(root, 'phase_category.txt')

        # phase_category (PHASE: phase) e.g. (LAM: lam)
        phase_cat: dict = {}
        with open(phase_category_file, 'r') as f:
            for line in f:
                ls = line.strip().split()
                phase_cat[ls[0]] = ls[1]

        if class_choice is not None:
            phase_cat = {k: v for k, v in phase_cat.items()
                         if k in class_choice}

        # (phase: PHASE) e.g. (lam: LAM)
        id2cat: dict = {v: k for k, v in phase_cat.items()}

        meta: dict = {}
        for item in phase_cat:
            meta[item] = []

        splitfile = os.path.join(
            root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))

        filelist = json.load(open(splitfile, 'r'))
        for file in filelist:
            _, category, uuid = file.split('/')
            if category in phase_cat.values():
                pts = os.path.join(root, category,
                                   'points', uuid + '.pts')
                meta[id2cat[category]].append(pts)

        self.datapath: list = []
        for item in phase_cat:
            for fn in meta[item]:
                self.datapath.append((item, fn))

        del meta

        self.classes: dict = dict(
            zip(sorted(phase_cat), range(len(phase_cat))))
        print("classes = {}".format(self.classes))

        if preprocess_data:
            if random_sample:
                save_path = os.path.join(
                    root, 'phase_%s_%d_pts.dat' % (split, npoints))
            else:
                save_path = os.path.join(
                    root, 'phase_%s_%d_pts_fps.dat' % (split, npoints))

            if os.path.exists(save_path):
                print('Load processed data from {}...'.format(save_path))
                with open(save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(
                        f)
            else:
                print('Processing data {} (only running in the first time)...'.format(
                    save_path))

                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]

                    cls = self.classes[fn[0]]
                    cls = np.array([cls]).astype(np.int32)
                    self.list_of_labels[index] = cls

                    points = np.loadtxt(fn[1]).astype(np.float32)

                    if random_sample:
                        tot_npoints = points.shape[0]
                        indices = np.random.choice(
                            tot_npoints, npoints, replace=(tot_npoints < npoints))
                        sampled_points = points[indices]
                    else:
                        sampled_points = sample_furthest_points(
                            points, npoints)

                    sampled_points = normalize_points(sampled_points)
                    if data_augmentation:
                        # random jitter
                        sampled_points += np.random.normal(
                            0, 0.02, size=sampled_points.shape)

                    self.list_of_points[index] = sampled_points

                with open(save_path, 'wb') as f:
                    pickle.dump(
                        [self.list_of_points, self.list_of_labels], f)

    def __getitem__(self, index: int):
        if self.preprocess_data:
            sampled_points = self.list_of_points[index]
            cls = self.list_of_labels[index]
        else:
            fn = self.datapath[index]

            cls = self.classes[fn[0]]
            cls = np.array([cls]).astype(np.int32)

            points = np.loadtxt(fn[1]).astype(np.float32)

            if self.random_sample:
                tot_npoints = points.shape[0]
                indices = np.random.choice(
                    tot_npoints, self.npoints, replace=(tot_npoints < self.npoints))
                sampled_points = points[indices]
            else:
                sampled_points = sample_furthest_points(points, self.npoints)

            sampled_points = normalize_points(sampled_points)

            if self.data_augmentation:
                # random jitter
                sampled_points += np.random.normal(
                    0, 0.02, size=sampled_points.shape)

        sampled_points = torch.from_numpy(sampled_points)
        cls = torch.from_numpy(cls.astype(np.int64))

        return sampled_points, cls

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    datapath = sys.argv[1]
    gen_phase_id(datapath)
    d = PhaseDataset(root=datapath)
    print(len(d))
    print(d[0])
