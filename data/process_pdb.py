import argparse
import MDAnalysis as mda
import numpy as np
import os
import time
from tqdm import tqdm


class Pdb2Pts(object):
    def __init__(self, category, trans, rot):

        self.category = category
        self.trans = trans
        self.rot = rot

    def apply_periodic_to_pos(self, pos, Lx, Ly, Lz):
        """
        Apply periodic boundary conditions
        """
        invLx, invLy, invLz = (1.0/Lx, 1.0/Ly, 1.0/Lz)
        X, Y, Z = pos[:, 0], pos[:, 1], pos[:, 2]
        scale_factor = np.floor(Z * invLz)
        Z -= scale_factor * Lz
        scale_factor = np.floor(Y * invLy)
        Y -= scale_factor * Ly
        scale_factor = np.floor(X * invLx)
        X -= scale_factor * Lx
        return pos

    def replicate_box(self, increment):
        """
        Input an increment array, e.g. [-1, 0, 1] for replicating the box three times in each dimension
        """
        combs = []
        for i in increment:
            for j in increment:
                for k in increment:
                    if not i == j == k == 0:
                        combs.append([i, j, k])
        return combs

    def rand_rotation_matrix(self):
        """
        Creates a random uniform rotation matrix.
        deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
        rotation. Small deflection => small perturbation.
        """
        # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

        randnums = np.random.uniform(size=(3,))

        theta, phi, z = randnums

        theta = theta * 2.0 * np.pi  # Rotation about the pole (Z).
        phi = phi * 2.0 * np.pi  # For direction of pole deflection.
        z = z * 2.0  # For magnitude of pole deflection.

        # Compute a vector V used for distributing points over the sphere
        # via the reflection I - V Transpose(V).  This formulation of V
        # will guarantee that if x[1] and x[2] are uniformly distributed,
        # the reflected points will be uniform on the sphere.  Note that V
        # has length sqrt(2) to eliminate the 2 in the Householder matrix.

        r = np.sqrt(z)
        V = (np.sin(phi) * r, np.cos(phi) * r, np.sqrt(2.0 - z))

        st = np.sin(theta)
        ct = np.cos(theta)

        R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

        # Construct the rotation matrix  ( V Transpose(V) - I ) R.

        M = (np.outer(V, V) - np.eye(3)).dot(R)

        return M

    def rand_periodic_translation(self, pos, Lx, Ly, Lz):
        """Randomly moving the atoms along the X, Y, and Z directions
        and apply the periodic boundary conditions on the new positions.
        """
        vec_trans = np.array([np.random.uniform(0, Lx * 0.5),
                              np.random.uniform(0, Ly * 0.5),
                              np.random.uniform(0, Lz * 0.5)])

        new_pos = pos + vec_trans
        new_pos = self.apply_periodic_to_pos(new_pos, Lx, Ly, Lz)

        return new_pos

    # pdb file must contain 1 frame
    def gen_pts_from_pdb(self, pdb_path, ntrans, npoints_min=1024):
        file_list = os.listdir(pdb_path)
        pdb_files = [i for i in file_list if 'pdb' in i]
        idx = 1
        # Create a log file for later debugging
        log_file = open(f'{opt.category}.log', 'w')
        for i in tqdm(range(1, len(pdb_files) + 1)):
            universe = mda.Universe(pdb_path + '/' + pdb_files[i - 1])
            oxygen = universe.select_atoms('name O')
            Lx, Ly, Lz = universe.dimensions[:3]

            orig_pos_oxygen = oxygen.positions

            # Make sure all atoms are within the periodic box
            orig_pos_oxygen = self.apply_periodic_to_pos(
                orig_pos_oxygen, Lx, Ly, Lz)

            log_file.write(pdb_files[i-1] + ', N=' + str(orig_pos_oxygen.shape[0]) +
                           ', Lx=' + str(Lx) + ', Ly=' + str(Ly) + ', Lz=' + str(Lz) + '\n')

            pos_oxygen = orig_pos_oxygen

            for i in range(ntrans):
                # Apply random data augmentation (translation + rotation) only for ntrans > 1
                # while preserving the original data instance
                while i > 0:
                    # Apply a random periodic translation around a vector with random x, y, and z components that are <= period
                    if self.trans:
                        pos_oxygen = self.rand_periodic_translation(
                            pos_oxygen, Lx, Ly, Lz)

                    # extra check to make sure the box contains enough O atoms
                    if pos_oxygen.shape[0] < npoints_min:
                        pos_oxygen = orig_pos_oxygen
                        continue

                    # Replicate box so that the transformed coordinates can be wrapped into the original bounding box
                    if self.rot:
                        orig_pos = pos_oxygen
                        increment = [-3, -2, -1, 0, 1, 2, 3]
                        for vec in self.replicate_box(increment):
                            pos_oxygen = np.vstack(
                                [pos_oxygen,  orig_pos + np.multiply([Lx, Ly, Lz], np.array(vec))])
                        M = self.rand_rotation_matrix()
                        pos_oxygen = np.dot(pos_oxygen, M)
                        pos_oxygen = pos_oxygen[np.logical_and(
                            pos_oxygen[:, 0] < Lx, pos_oxygen[:, 0] >= 0)]
                        pos_oxygen = pos_oxygen[np.logical_and(
                            pos_oxygen[:, 1] < Ly, pos_oxygen[:, 1] >= 0)]
                        pos_oxygen = pos_oxygen[np.logical_and(
                            pos_oxygen[:, 2] < Lz, pos_oxygen[:, 2] >= 0)]

                    # extra check to make sure the box contains enough O atoms
                    if pos_oxygen.shape[0] < npoints_min:
                        pos_oxygen = orig_pos_oxygen
                        continue

                    break

                file_name = f'coord_O_{opt.category}_{idx}'
                idx += 1

                log_file.write(
                    str(pos_oxygen.shape[0]) + ' ' + file_name + '\n')

                # Save .pts files
                pts_file = open('point_clouds/' + opt.category +
                                '/points/' + file_name + '.pts', 'w')
                for k in range(pos_oxygen.shape[0]):
                    pts_file.write(str(
                        pos_oxygen[k, 0]) + ' ' + str(pos_oxygen[k, 1]) + ' ' + str(pos_oxygen[k, 2]) + '\n')
                pts_file.close()

                # Save .xyz files
                xyz_file = open('point_clouds/' + opt.category +
                                '/xyz/' + file_name + '.xyz', 'w')
                xyz_file.write(str(pos_oxygen.shape[0]) + '\n\n')
                for k in range(pos_oxygen.shape[0]):
                    xyz_file.write('O  ' + str(pos_oxygen[k, 0]) + ' ' + str(
                        pos_oxygen[k, 1]) + ' ' + str(pos_oxygen[k, 2]) + '\n')
                xyz_file.close()

                # Save .seg files
                seg_file = open('point_clouds/' + opt.category +
                                '/points_label/' + file_name + '.seg', 'w')
                for k in range(pos_oxygen.shape[0]):
                    seg_file.write('1\n')
                seg_file.close()

        log_file.close()

        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--category', type=str, required=True,
                        help='category (phase) name: lam (lamellar), hpc (hexagonally packed cylinder), hpl (hexagonally perforated lamellar), bcc (body-centered cubic), dis (disordered states), sg (single gyroid), dg (double gyroid), dd (double diamond), p (plumber\'s nightmare')
    parser.add_argument('-t', '--rand_trans', action='store_true',
                        help='whether to apply random periodic translation')
    parser.add_argument('-r', '--rand_rot', action='store_true',
                        help='whether to apply random periodic rotation')
    parser.add_argument('-nt',
                        '--ntrans', type=int, default=1, help='number of random data augmentation (translation + rotation) for each point cloud')
    parser.add_argument('-np',
                        '--npoints_min', type=int, default=1024, help='minimum number of points that the augnmented box must contain')
    opt = parser.parse_args()
    opt.category = opt.category.lower()

    if not os.path.exists(os.path.join('point_clouds', opt.category, 'points')):
        os.makedirs(os.path.join('point_clouds', opt.category, 'points'))
    if not os.path.exists(os.path.join('point_clouds', opt.category, 'xyz')):
        os.makedirs(os.path.join('point_clouds', opt.category, 'xyz'))
    if not os.path.exists(os.path.join('point_clouds', opt.category, 'points_label')):
        os.makedirs(os.path.join('point_clouds', opt.category, 'points_label'))

    pdb_path = os.path.join('raw', 'pdb', opt.category)

    print('Processing pdb files of %s structure...' % opt.category)

    print(opt.rand_trans)
    print(opt.rand_rot)

    tic = time.perf_counter()
    Pdb2Pts(opt.category, opt.rand_trans, opt.rand_rot).gen_pts_from_pdb(
        pdb_path, opt.ntrans, opt.npoints_min)
    toc = time.perf_counter()
    print('Used %d ms.' % ((toc - tic) * 1000))
