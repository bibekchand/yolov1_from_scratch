import torch
import torch.nn as nn

C = 19  # number of classes
B = 2   # number of boxes per grid
S = 7   # grid size


def giou(box1, box2, in_format=False):
    """
    Generalized IoU
    box1, box2: (..., 4) -> [x, y, w, h] if in_format=False
    """
    eps = 1e-6
    if in_format:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]
    else:
        # Convert [x, y, w, h] -> [x1, y1, x2, y2]
        b1_x1, b1_y1 = box1[..., 0] - box1[..., 2]/2, box1[..., 1] - box1[..., 3]/2
        b1_x2, b1_y2 = box1[..., 0] + box1[..., 2]/2, box1[..., 1] + box1[..., 3]/2
        b2_x1, b2_y1 = box2[..., 0] - box2[..., 2]/2, box2[..., 1] - box2[..., 3]/2
        b2_x2, b2_y2 = box2[..., 0] + box2[..., 2]/2, box2[..., 1] + box2[..., 3]/2

    # Intersection
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    # Union
    b1_area = (b1_x2 - b1_x1).clamp(0) * (b1_y2 - b1_y1).clamp(0)
    b2_area = (b2_x2 - b2_x1).clamp(0) * (b2_y2 - b2_y1).clamp(0)
    union_area = b1_area + b2_area - inter_area + eps

    iou = inter_area / union_area

    # Convex hull
    hull_x1 = torch.min(b1_x1, b2_x1)
    hull_y1 = torch.min(b1_y1, b2_y1)
    hull_x2 = torch.max(b1_x2, b2_x2)
    hull_y2 = torch.max(b1_y2, b2_y2)
    hull_area = (hull_x2 - hull_x1).clamp(0) * (hull_y2 - hull_y1).clamp(0) + eps

    return iou - (hull_area - union_area)/hull_area


def loss(prediction, target):
    """
    prediction: [batch, 7, 7, C + 5*B] -> [classes, conf1, box1, conf2, box2]
    target: [batch, 7, 7, C + 1 + 4] -> [classes, obj, box]
    """
    sse = nn.MSELoss(reduction="sum")
    lambda_coord = 5
    lambda_noobj = 0.5
    eps = 1e-7

    # --------------------
    # Masks
    # --------------------
    obj = target[..., C:C+1]          # object mask
    noobj = 1 - obj
    target_bbox = target[..., C+1:C+5]

    # --------------------
    # Prediction components
    # --------------------
    pred_conf0 = prediction[..., C:C+1]
    pred_bbox0 = prediction[..., C+1:C+5]
    pred_conf1 = prediction[..., C+5:C+6]
    pred_bbox1 = prediction[..., C+6:C+10]

    # --------------------
    # GIoU
    # --------------------
    giou0 = giou(pred_bbox0, target_bbox, in_format=False).unsqueeze(-1)
    giou1 = giou(pred_bbox1, target_bbox, in_format=False).unsqueeze(-1)
    gious = torch.cat([giou0, giou1], dim=-1)
    best_giou, best_idx = torch.max(gious, dim=-1, keepdim=True)

    # --------------------
    # Responsible bbox
    # --------------------
    best_bbox = (1 - best_idx) * pred_bbox0 + best_idx * pred_bbox1

    # Sqrt trick
    target_wh = torch.sqrt(target_bbox[..., 2:4] + eps)
    pred_wh = torch.sign(best_bbox[..., 2:4]) * torch.sqrt(torch.abs(best_bbox[..., 2:4]) + eps)

    # --------------------
    # Loss components
    # --------------------
    bbox_loss = sse(obj * best_bbox[..., :2], obj * target_bbox[..., :2]) + \
                sse(obj * pred_wh, obj * target_wh)

    best_conf = (1 - best_idx) * pred_conf0 + best_idx * pred_conf1
    object_loss = sse(obj * best_conf, obj * best_giou.detach())

    no_object_loss = sse(noobj * pred_conf0, noobj * obj) + \
                     sse(noobj * pred_conf1, noobj * obj)

    class_loss = sse(obj * prediction[..., :C], obj * target[..., :C])

    # --------------------
    # Total
    # --------------------
    total_loss = lambda_coord * bbox_loss + object_loss + lambda_noobj * no_object_loss + class_loss

    return total_loss
