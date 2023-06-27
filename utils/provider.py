import os
import plyfile
import torch
import numpy as np


color_palette = [(60,  179, 113), (239, 177,  24), (70,  130, 180), (255, 127,  80), (0,   191, 255), (138,  43, 226),
                 (225,  40,  40), (40,  225,  40), (40,   40, 225), (225, 225,   0), (64,  224, 208), (225,   0, 225),
                 (192, 192, 192), (128,   0,   0), (128, 128,   0), (0,   128,   0), (128,   0, 128), (0,   128, 128),
                 (0,     0, 128), (165,  42,  42), (100, 149, 237), (233, 150, 122), (255, 140,   0), (240, 197, 117),
                 (189, 183, 107), (154, 205,  50), (85,  107,  47), (34,  139,  34), (152, 251, 152), (0,   250, 154),
                 (75,    0, 130), (218, 112, 214), (0,   225, 225), (139,  69,  19), (188, 143, 143), (112, 128, 144)]


def normalize_data(batch_data):
    """ Normalize the batch data, use coordinates of the block centered at origin,
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    B, N, C = batch_data.shape
    normal_data = np.zeros((B, N, C))
    for b in range(B):
        pc = batch_data[b]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        normal_data[b] = pc
    return normal_data


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def shuffle_points(batch_data):
    """ Shuffle orders of points in each point cloud -- changes FPS behavior.
        Use the same shuffling idx for the entire batch.
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    idx = np.arange(batch_data.shape[1])
    np.random.shuffle(idx)
    return batch_data[:, idx, :]


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_z(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_with_normal(batch_xyz_normal):
    """ Randomly rotate XYZ, normal point cloud.
        Input:
            batch_xyz_normal: B,N,6, first three channels are XYZ, last 3 all normal
        Output:
            B,N,6, rotated XYZ, normal point cloud
    """
    for k in range(batch_xyz_normal.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_xyz_normal[k, :, 0:3]
        shape_normal = batch_xyz_normal[k, :, 3:6]
        batch_xyz_normal[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        batch_xyz_normal[k, :, 3:6] = np.dot(shape_normal.reshape((-1, 3)), rotation_matrix)
    return batch_xyz_normal


def rotate_perturbation_point_cloud_with_normal(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx6 array, original batch of point clouds and point normals
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        shape_pc = batch_data[k, :, 0:3]
        shape_normal = batch_data[k, :, 3:6]
        rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), R)
        rotated_data[k, :, 3:6] = np.dot(shape_normal.reshape((-1, 3)), R)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        # rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, :, 0:3]
        rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle_with_normal(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx6 array, original batch of point clouds with normal
          scalar, angle of rotation
        Return:
          BxNx6 array, rotated batch of point clouds iwth normal
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        # rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, :, 0:3]
        shape_normal = batch_data[k, :, 3:6]
        rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        rotated_data[k, :, 3:6] = np.dot(shape_normal.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert (clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data


def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B, 3))
    for batch_index in range(B):
        batch_data[batch_index, :, :] += shifts[batch_index, :]
    return batch_data


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index, :, :] *= scales[batch_index]
    return batch_data


def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    """ batch_pc: BxNx3 """
    for b in range(batch_pc.shape[0]):
        dropout_ratio = np.random.random() * max_dropout_ratio  # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            batch_pc[b, drop_idx, :] = batch_pc[b, 0, :]  # set to the first point
    return batch_pc


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def save_ply(points, filename, normals=None, colors=None):
    vertex = np.core.records.fromarrays(np.array(points).transpose(), names='x, y, z', formats='f4, f4, f4')
    desc = vertex.dtype.descr

    if normals is not None:
        assert len(normals) == len(points)
        vertex_normal = np.core.records.fromarrays(np.array(normals).transpose(), names='nx, ny, nz', formats='f4, f4, f4')
        desc = desc + vertex_normal.dtype.descr

    if colors is not None:
        assert len(colors) == len(points)
        vertex_color = np.core.records.fromarrays(np.array(colors).transpose() * 255.0, names='red, green, blue', formats='u1, u1, u1')
        desc = desc + vertex_color.dtype.descr

    vertex_all = np.empty(len(points), dtype=desc)

    for prop in vertex.dtype.names:
        vertex_all[prop] = vertex[prop]

    if normals is not None:
        for prop in vertex_normal.dtype.names:
            vertex_all[prop] = vertex_normal[prop]

    if colors is not None:
        for prop in vertex_color.dtype.names:
            vertex_all[prop] = vertex_color[prop]

    ply = plyfile.PlyData([plyfile.PlyElement.describe(vertex_all, 'vertex')], text=False)
    if os.path.dirname(filename) != '' and not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    ply.write(filename)


def save_ply_patches(points, filename, normals=None, colors=None):
    vertex = np.core.records.fromarrays(np.concatenate(points, axis=0).transpose(), names='x, y, z', formats='f4, f4, f4')
    desc = vertex.dtype.descr

    if normals is not None:
        vertex_normal = []
        for i in range(points.shape[0]):
            assert len(normals[i]) == len(points[i])
            vertex_normal.append(normals[i])
        vertex_normal = np.concatenate(vertex_normal, axis=0)
        vertex_normal = np.core.records.fromarrays(np.array(vertex_normal).transpose(), names='nx, ny, nz', formats='f4, f4, f4')
        desc = desc + vertex_normal.dtype.descr

    if colors is not None:
        vertex_color = []
        for i in range(points.shape[0]):
            assert len(colors[i]) == len(points[i])
            vertex_color.append(colors[i])
        vertex_color = np.concatenate(vertex_color, axis=0)
        vertex_color = np.core.records.fromarrays(np.array(vertex_color).transpose() * 255.0, names='red, green, blue', formats='u1, u1, u1')
        desc = desc + vertex_color.dtype.descr

    vertex_all = np.empty(len(np.concatenate(points, axis=0)), dtype=desc)

    for prop in vertex.dtype.names:
        vertex_all[prop] = vertex[prop]

    if normals is not None:
        for prop in vertex_normal.dtype.names:
            vertex_all[prop] = vertex_normal[prop]

    if colors is not None:
        for prop in vertex_color.dtype.names:
            vertex_all[prop] = vertex_color[prop]

    ply = plyfile.PlyData([plyfile.PlyElement.describe(vertex_all, 'vertex')], text=False)
    if os.path.dirname(filename) != '' and not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    ply.write(filename)


def save_ply_batch(points_batch, file_path, names=None, step=0, patch=False):
    batch_size = len(points_batch)

    # if type(file_path) != list:
    #     basename = os.path.splitext(file_path)[0]
    #     ext = '.ply'

    if patch:
        colors = []
        for i in range(points_batch.shape[1]):
            color_i = np.repeat([color_palette[i]], points_batch.shape[2], axis=0)
            colors.append(color_i)
        colors = np.array(colors) / 255.0

    for batch_idx in range(batch_size):
        if patch:
            if names is None:
                save_name = os.path.join(file_path[batch_idx] if type(file_path) == list else file_path,
                                         '%04d_patches.ply' % (step * batch_size + batch_idx))
            else:
                save_name = os.path.join(file_path[batch_idx] if type(file_path) == list else file_path,
                                         names[batch_idx] + '.ply')

            save_ply_patches(points_batch[batch_idx], save_name, colors=colors)
        else:
            if names is None:
                save_name = os.path.join(file_path[batch_idx] if type(file_path) == list else file_path,
                                         '%04d.ply' % (step * batch_size + batch_idx))
            else:
                save_name = os.path.join(file_path[batch_idx] if type(file_path) == list else file_path,
                                         names[batch_idx] + '.ply')

            save_ply(points_batch[batch_idx], save_name)

        # if type(file_path) == list:
        #     save_ply(points_batch[batch_idx], os.path.join(file_path[batch_idx], '%04d.ply' % (step * batch_size + batch_idx)))
        # else:
        #     save_ply(points_batch[batch_idx], os.path.join(file_path, '%04d.ply' % (step * batch_size + batch_idx)))


def load_data_id(path):
    fopen = open(path, 'r', encoding='utf-8')
    lines = fopen.readlines()
    id_list = []
    linecount = 0

    for line in lines:
        if line == '\n':
            continue
        id_list.append(line.strip('\n'))
        linecount = linecount + 1
    fopen.close()
    return id_list


# def check_and_create_dirs(dir_list):
#     for dir in dir_list:
#         if not os.path.exists(dir):
#             os.makedirs(dir)
#             print(dir + ' does not exist. Created.')


def save_off_points(points, path):
    with open(path, "w") as file:
        file.write("OFF\n")
        file.write(str(int(points.shape[0])) + " 0" + " 0\n")
        for i in range(points.shape[0]):
            file.write(
                str(float(points[i][0])) + " " + str(float(points[i][1])) + " " + str(float(points[i][2])) + "\n")


def save_off_mesh(v, f, path):
    with open(path, "w") as file:
        file.write("OFF\n")
        v_num = len(v)
        f_num = len(f)
        file.write(str(v_num) + " " + str(len(f)) + " " + str(0) + "\n")
        for j in range(v_num):
            file.write(str(float(v[j][0])) + " " + str(float(v[j][1])) + " " + str(float(v[j][2])) + "\n")
        for j in range(f_num):
            file.write("3 " + str(int(f[j][0])) + " " + str(int(f[j][1])) + " " + str(int(f[j][2])) + "\n")


def save_coff_points(points, colors, path):
    with open(path, "w") as file:
        file.write("COFF\n")
        file.write(str(int(points.shape[0])) + " 0" + " 0\n")
        for i in range(points.shape[0]):
            file.write(str(float(points[i][0])) + " " + str(float(points[i][1])) + " " + str(float(points[i][2])) + " ")
            file.write(str(colors[i][0]) + " " + str(colors[i][1]) + " " + str(colors[i][2]) + "\n")


def save_graph(v, A, path):
    with open(path, "w") as file:
        file.write("g line\n")
        v_num = len(v)
        for j in range(v_num):
            file.write("v " + str(float(v[j][0])) + " " + str(float(v[j][1])) + " " + str(float(v[j][2])) + "\n")
        file.write("g\n")

        # A is a symmetric matrix
        for j in range(v_num):
            for k in range(j + 1, v_num):
                if A[j][k] == 1:
                    file.write("l " + str(j + 1) + " " + str(k + 1) + "\n")


def save_spheres(center, radius, path):
    sp_v, sp_f = load_off('./utils/sphere16.off')

    with open(path, "w") as file:
        for i in range(center.shape[0]):
            v, r = center[i], radius[i]
            v_ = sp_v * r
            v_ = v_ + v
            for m in range(v_.shape[0]):
                file.write('v ' + str(v_[m][0]) + ' ' + str(v_[m][1]) + ' ' + str(v_[m][2]) + '\n')

        for m in range(center.shape[0]):
            base = m * sp_v.shape[0] + 1
            for j in range(sp_f.shape[0]):
                file.write(
                    'f ' + str(sp_f[j][0] + base) + ' ' + str(sp_f[j][1] + base) + ' ' + str(sp_f[j][2] + base) + '\n')


def save_skel_mesh(v, f, e, path_f, path_e):
    f_file = open(path_f, "w")
    e_file = open(path_e, "w")
    v_num = len(v)
    f_num = len(f)
    e_num = len(e)

    for j in range(v_num):
        f_file.write('v ' + str(float(v[j][0])) + " " + str(float(v[j][1])) + " " + str(float(v[j][2])) + "\n")
    for j in range(f_num):
        f_file.write("f " + str(int(f[j][0]) + 1) + " " + str(int(f[j][1]) + 1) + " " + str(int(f[j][2]) + 1) + "\n")

    for j in range(v_num):
        e_file.write('v ' + str(float(v[j][0])) + " " + str(float(v[j][1])) + " " + str(float(v[j][2])) + "\n")
    for j in range(e_num):
        e_file.write("l " + str(int(e[j][0]) + 1) + " " + str(int(e[j][1]) + 1) + "\n")

    f_file.close()
    e_file.close()


def save_skel_xyzr(v, r, path):
    file = open(path, "w")
    v_num = len(v)
    file.write(str(v_num) + "\n")
    for i in range(v_num):
        file.write(
            str(float(v[i][0])) + " " + str(float(v[i][1])) + " " + str(float(v[i][2])) + " " + str(float(r[i])) + "\n")
    file.close()


def save_colored_weights(path, shape_name, weights, samples):
    skel_num = weights.shape[0]
    sample_num = weights.shape[1]
    min_gray = 200
    for i in range(skel_num):
        colors = np.zeros((sample_num, 3)).astype(np.int)
        max_w = max(weights[i].tolist())
        for j in range(sample_num):
            color = min_gray - int((weights[i][j] / max_w) * min_gray)
            colors[j] = np.array([color, color, color], np.int)

        save_coff_points(samples, colors, path + str(shape_name) + '_' + str(i) + '_weight.off')


def load_off(path):
    fopen = open(path, 'r', encoding='utf-8')
    lines = fopen.readlines()
    linecount = 0
    pts = np.zeros((1, 3), np.float64)
    faces = np.zeros((1, 3), np.int)
    p_num = 0
    f_num = 0

    for line in lines:
        linecount = linecount + 1
        word = line.split()

        if linecount == 1:
            continue
        if linecount == 2:
            p_num = int(word[0])
            f_num = int(word[1])
            pts = np.zeros((p_num, 3), np.float)
            faces = np.zeros((f_num, 3), np.int)
        if 3 <= linecount < 3 + p_num:
            pts[linecount - 3, :] = np.float64(word[0:3])
        if linecount >= 3 + p_num:
            faces[linecount - 3 - p_num] = np.int32(word[1:4])

    fopen.close()
    return pts, faces


def load_ply_points(pc_filepath, expected_point=2000):
    fopen = open(pc_filepath, 'r', encoding='utf-8')
    lines = fopen.readlines()
    pts = np.zeros((expected_point, 3), np.float64)

    total_point = 0
    feed_point_count = 0

    start_point_data = False
    for line in lines:
        word = line.split()
        if word[0] == 'element' and word[1] == 'vertex':
            total_point = int(word[2])
            # if expected_point > total_point:
            #     pts = np.zeros((total_point, 3), np.float64)
            # continue

        if start_point_data:
            pts[feed_point_count, :] = np.float64(word[0:3])
            feed_point_count += 1

        if word[0] == 'end_header':
            start_point_data = True

        if feed_point_count >= expected_point:
            break

    fopen.close()
    return pts


def linspace_nd(start, end, steps=10):
    """
    Vectorized version of torch.linspace.
    Inputs:
    - start: Tensor of any shape
    - end: Tensor of the same shape as start
    - steps: Integer
    Returns:
    - out: Tensor of shape start.size() + (steps,), such that
      out.select(-1, 0) == start, out.select(-1, -1) == end,
      and the other elements of out linearly interpolate between
      start and end.
    """
    assert start.size() == end.size()

    wgt_size = start.dim() * (1,) + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps).to(start)
    start_w = start_w.view(wgt_size).expand(out_size)

    end_w = torch.linspace(0, 1, steps=steps).to(end)
    end_w = end_w.view(wgt_size).expand(out_size)

    start = start.contiguous().unsqueeze(-1).expand(out_size)
    end = end.contiguous().unsqueeze(-1).expand(out_size)

    out = start_w * start + end_w * end

    return out


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids
