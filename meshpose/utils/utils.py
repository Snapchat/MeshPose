import cv2
import numpy as np
import pycocotools.mask as mask_util


def gen_trans_from_patch_cv(c_x, c_y, src_w, src_h, dst_w, dst_h):

    src_center = np.array([c_x, c_y], dtype=np.float32)
    src_downdir = np.array([0, src_h * 0.5], dtype=np.float32)
    src_rightdir = np.array([src_w * 0.5, 0], dtype=np.float32)
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    trans_inv = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans, trans_inv


def framebbox(bbox, aspect_ratio, scale_bbox=1.0):
    x, y, w, h = bbox
    center_x = x + w * 0.5
    center_y = y + h * 0.5

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    w = w * scale_bbox
    h = h * scale_bbox

    return center_x, center_y, w, h


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.0]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def compute_transform_jacobian(t):
    return t[0, 0] * t[1, 1] - t[0, 1] * t[1, 0]


def affine_tranform_3d(points, transform):
    jacobian = compute_transform_jacobian(transform)
    scale = np.sqrt(np.abs(jacobian))
    for i in range(points.shape[0]):
        points[i, 0:2] = affine_transform(points[i, 0:2], transform)
        points[i, 2] *= scale
    return points


def read_dp_mask(polys):
    mask = np.zeros([256, 256])
    for i in range(1, 15):
        if polys[i-1]:
            current_mask = mask_util.decode(polys[i - 1])
            mask[current_mask > 0] = i
    return mask


def imread(path):
    return cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)[..., :3]


def imwrite(path, image):
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def round_np(array, decimals=4):
    return np.round(array.tolist(), decimals)


def visualize_vertices(image, results, vertices_type):
    for result in results:
        vertices = result[vertices_type]
        for vertex in vertices:
            coords = (int(vertex[0]), int(vertex[1]))
            cv2.circle(image, coords, radius=5, color=(0, 255, 0), thickness=-1)
    return image
