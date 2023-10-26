import numpy as np

def rect_iou(rect1, rect2):
    
    assert rect1.shape == rect2.shape
    rects_inter = _intersection(rect1, rect2)
    areas_inter = np.prod(rects_inter[..., 2:], axis=1)
    
    areas1 = np.prod(rect1[..., 2:], axis=-1)
    areas2 = np.prod(rect2[..., 2:], axis=-1)
    areas_union = areas1 + areas2 - areas_inter

    ious = areas_inter / areas_union
    ious = np.clip(ious, 0.0, 1.0)

    return ious

def _intersection(rect1, rect2):
    """Rectangle intersection"""
    assert rect1.shape == rect2.shape
    x1 = np.maximum(rect1[..., 0], rect2[..., 0])
    y1 = np.maximum(rect1[..., 1], rect2[..., 1])
    x2 = np.minimum(rect1[..., 0] + rect1[..., 2],
                    rect2[..., 0] + rect2[..., 2])
    y2 = np.minimum(rect1[..., 1] + rect1[..., 3],
                    rect2[..., 1] + rect2[..., 3])
    
    w = np.maximum(x2 - x1, 0)
    h = np.maximum(y2 - y1, 0)

    return np.stack([x1, y1, w, h]).T

def evaluate(ious):
    ao = np.mean(ious) * 100            # area overlap
    sr_05 = np.mean(ious > 0.5) * 100   # success rate when iou > 0.5
    sr_075 = np.mean(ious > 0.75) * 100 # success rate when iou > 0.75 
    return ao, sr_05, sr_075

def center_error(rect1, rect2):
    centers1 = rect1[..., :2] + (rect1[..., 2:] - 1) / 2
    centers2 = rect2[..., :2] + (rect2[..., 2:] - 1) / 2
    errors = np.sqrt(np.sum(np.power(centers1 - centers2, 2), axis=-1))
    return errors

def normal_center_error(rect1, rect2):
    centers1 = rect1[..., :2] + (rect1[..., 2:] - 1) / 2
    centers2 = rect2[..., :2] + (rect2[..., 2:] - 1) / 2
    errors = np.sqrt(np.sum(np.power((centers1 - centers2)/np.maximum(np.array([[1.,1.]]), rect2[:, 2:]), 2), axis=-1))
    return errors

def calc_curves(ious, center_errors, norm_center_errors):
    ious = np.asarray(ious, float)[:, np.newaxis]
    center_errors = np.asarray(center_errors, float)[:, np.newaxis]
    norm_center_errors = np.asarray(norm_center_errors, float)[:, np.newaxis]

    n = 21
    thr_iou = np.linspace(0, 1, n)[np.newaxis, :]
    thr_ce = np.arange(0, 51)[np.newaxis, :]
    thr_nce = np.linspace(0, 0.5, 51)[np.newaxis, :]
    
    bin_iou = np.greater(ious, thr_iou)
    bin_ce = np.less_equal(center_errors, thr_ce)
    bin_nce = np.less_equal(norm_center_errors, thr_nce)

    succ_curve = np.mean(bin_iou, axis=0)
    prec_curve = np.mean(bin_ce, axis=0)
    norm_prec_curve = np.mean(bin_nce, axis=0)

    return succ_curve, prec_curve, norm_prec_curve

def get_scores(gt_path, tr_path):
    gt_boxes = np.loadtxt(gt_path, delimiter=',')
    tr_boxes = np.loadtxt(tr_path, delimiter=',')
    assert (gt_boxes.shape == tr_boxes.shape)

    # Calculate IoU to find AO and SR for GOT-10k
    seq_ious = rect_iou(tr_boxes, gt_boxes)
    ao, sr_05, sr_075 = evaluate(seq_ious)
    print("--- Evaluate for GOT-10k ---")
    print("Average Overlap (AO): {:.2f} %".format(ao))
    print("Success 0.5 (SR0.5): {:.2f} %".format(sr_05))
    print("Success 0.75 (SR0.75): {:.2f} %".format(sr_075))

    # Calculate center error for LaSOT, TrackingNet
    cen_err = center_error(tr_boxes, gt_boxes)
    ncen_err = normal_center_error(tr_boxes, gt_boxes)
    succ_curve, prec_curve, norm_prec_curve = calc_curves(seq_ious, cen_err, ncen_err)
    print("--- Evaluate for TrackingNet & LaSOT ---")
    succ_score = np.mean(succ_curve) * 100
    prec_score = prec_curve[20] * 100
    norm_prec_score = np.mean(norm_prec_curve) * 100
    print("Success score (AUC): {:.2f} %".format(succ_score))
    print("Precision score (P): {:.2f} %".format(prec_score))
    print("NPrecision score (P_norm): {:.2f} %".format(norm_prec_score))