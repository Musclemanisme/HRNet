'''
function: load the checkpoint and evaluaton
'''

# initial the model

import torch
import PIL.Image as Image
from torchvision import transforms
import os
# from utils import ext_transforms as et
# import network
import tqdm
import numpy as np
from matplotlib import pyplot


def validate(opts, model, loader, device):
    """Do validation and return specified samples"""
    J = []
    F = []

    with torch.no_grad():
        for i, (images, labels, names) in enumerate(loader):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            # print("image:",images.size(),'labels:',labels.size(),'names:',names)
            outputs = model(images)

            preds = outputs.detach().max(dim=1)[1].cpu()
            for label, pred, name in zip(labels, preds, names):
                ##saveImage
                name = name.split('/')[-1]
                label = label.cpu().numpy()
                pred = pred.numpy()
                pyplot.imsave(os.path.join('eval_output', name), pred)

                J.append(db_eval_iou(label, pred))
                F.append(db_eval_boundary(pred, label))
                print('F:', sum(F) / len(F), 'J:', sum(J) / len(J))
        F = sum(F) / len(F)
        J = sum(J) / len(J)
        import logging
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler('./eval_output/metrics.log', 'w')
        fh.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")

        fh.setFormatter(formatter)

        logger.addHandler(fh)

        logger.info('F:%f' % F)
        logger.info('J;%f' % J)
        return F, J



def db_eval_iou(annotation, segmentation):
    """ Compute region similarity as the Jaccard Index.
    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.
    Return:
        jaccard (float): region similarity
    """

    annotation = annotation.astype(np.bool)
    segmentation = segmentation.astype(np.bool)

    if np.isclose(np.sum(annotation), 0) and np.isclose(np.sum(segmentation), 0):
        return 1
    else:
        return np.sum((annotation & segmentation)) / np.sum((annotation | segmentation), dtype=np.float32)


def db_eval_boundary(foreground_mask, gt_mask, bound_th=0.008):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.

    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.

    Returns:
        F (float): boundaries F-measure
        P (float): boundaries precision
        R (float): boundaries recall
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1

    bound_pix = bound_th if bound_th >= 1 else \
        np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = seg2bmap(foreground_mask);
    gt_boundary = seg2bmap(gt_mask);

    from skimage.morphology import binary_dilation, disk

    fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    gt_dil = binary_dilation(gt_boundary, disk(bound_pix))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall);

    return F


def seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.

    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]

    Returns:
        bmap (ndarray):	Binary boundary map.

     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
    """

    seg = seg.astype(np.bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (width > w | height > h | abs(ar1 - ar2) > 0.01), \
        'Can''t convert %dx%d seg to %dx%d bmap.' % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + floor((y - 1) + height / h)
                    i = 1 + floor((x - 1) + width / h)
                    bmap[j, i] = 1;

    return bmap


if __name__ == '__main__':
    from torch.utils import data
    from main import get_argparser, get_dataset

    opts = get_argparser().parse_args()
    opts.path_txt = {'images_train': '/raid/dataset/cvpr/images_train.txt',
                     'labels_train': '/raid/dataset/cvpr/labels_train.txt',
                     'images_val': '/raid/dataset/cvpr/images_val.txt',
                     'labels_val': '/raid/dataset/cvpr/labels_val.txt',
                     }
    opts.num_classes = 18

    _, val_dst = get_dataset(opts)

    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=0, drop_last=True)

    model = network.deeplabv3plus_resnet101(num_classes=opts.num_classes, output_stride=opts.output_stride)
    para_dict = torch.load('./checkpoints/latest_deeplabv3plus_resnet101_cvpr_os16.pth')
    model.load_state_dict(para_dict['model_state'])

    # net_str = str(net)
    # import pdb
    # pdb.set_trace()

    # with open('net.txt',"w") as f:
    #     f.write(net_str)
    # os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda:0')

    model = model.to(device)

    val_score, ret_samples = validate(
        opts=opts, model=model, loader=val_loader, device=device)



