import torch
import torch.nn as nn


def giou(box1, box2, in_format=False):
    """
    Calculates Generalized IoU.
    in_format=True: [x1, y1, x2, y2]
    in_format=False: [x, y, w, h]
    """
    eps = 1e-6
    if in_format:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[...,
                                          0], box1[..., 1], box1[..., 2], box1[..., 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[...,
                                          0], box2[..., 1], box2[..., 2], box2[..., 3]
    else:
        # [x, y, w, h] to [x1, y1, x2, y2]
        b1_x1, b1_y1 = box1[..., 0] - box1[..., 2] / \
            2, box1[..., 1] - box1[..., 3]/2
        b1_x2, b1_y2 = box1[..., 0] + box1[..., 2] / \
            2, box1[..., 1] + box1[..., 3]/2
        b2_x1, b2_y1 = box2[..., 0] - box2[..., 2] / \
            2, box2[..., 1] - box2[..., 3]/2
        b2_x2, b2_y2 = box2[..., 0] + box2[..., 2] / \
            2, box2[..., 1] + box2[..., 3]/2

    # Intersection
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    intersection_area = (inter_x2 - inter_x1).clamp(0) * \
        (inter_y2 - inter_y1).clamp(0)

    # Union
    b1_area = (b1_x2 - b1_x1).clamp(0) * (b1_y2 - b1_y1).clamp(0)
    b2_area = (b2_x2 - b2_x1).clamp(0) * (b2_y2 - b2_y1).clamp(0)
    union_area = b1_area + b2_area - intersection_area + eps

    iou = intersection_area / union_area

    # Convex Hull (Enclosing Box)
    hull_x1 = torch.min(b1_x1, b2_x1)
    hull_y1 = torch.min(b1_y1, b2_y1)
    hull_x2 = torch.max(b1_x2, b2_x2)
    hull_y2 = torch.max(b1_y2, b2_y2)
    convex_hull_area = (hull_x2 - hull_x1).clamp(0) * \
        (hull_y2 - hull_y1).clamp(0) + eps

    return iou - (convex_hull_area - union_area) / convex_hull_area


def loss(prediction, target):
    """
    Args:
        prediction: (batch, 7, 7, 30) -> [classes(20), conf1, x1, y1, w1, h1, conf2, x2, y2, w2, h2]
        target: (batch, 7, 7, 25) -> [classes(20), conf, x, y, w, h]
    """
    sse = nn.MSELoss(reduction="sum")
    lambda_coord = 5
    lambda_noobj = 0.5
    eps = 1e-7

    # 1. Target Masking
    # Ground truth: Class scores [0:20], Confidence [20], Box [21:25]
    obj = target[..., 20:21]  # (N, 7, 7, 1)
    noobj = 1 - obj
    target_bbox = target[..., 21:25]

    # 2. Extract Prediction Components
    # Prediction: Class [0:20], Box1_Conf [20], Box1_Coords [21:25], Box2_Conf [25], Box2_Coords [26:30]
    pred_conf0 = prediction[..., 20:21]
    pred_bbox0 = prediction[..., 21:25]
    pred_conf1 = prediction[..., 25:26]
    pred_bbox1 = prediction[..., 26:30]

    # 3. Calculate GIoU for Box Selection
    giou0 = giou(pred_bbox0, target_bbox, in_format=False).unsqueeze(-1)
    giou1 = giou(pred_bbox1, target_bbox, in_format=False).unsqueeze(-1)

    # Cat GIoU scores to find the "best" box (responsible box)
    gious = torch.cat([giou0, giou1], dim=-1)
    best_giou, best_idx = torch.max(gious, dim=-1, keepdim=True)

    # 4. Coordinate Loss
    # Logic: If best_idx is 0, pick pred_bbox0. If 1, pick pred_bbox1.
    best_bbox = (1 - best_idx) * pred_bbox0 + best_idx * pred_bbox1

    # YOLO sqrt trick: stabilizes loss for different box sizes
    target_wh = torch.sqrt(target_bbox[..., 2:4] + eps)
    pred_wh = torch.sign(best_bbox[..., 2:4]) * \
        torch.sqrt(torch.abs(best_bbox[..., 2:4]) + eps)

    bbox_loss = sse(obj * best_bbox[..., :2], obj * target_bbox[..., :2]) + \
        sse(obj * pred_wh, obj * target_wh)

    # 5. Object Confidence Loss (Target is the GIoU score)
    best_conf = (1 - best_idx) * pred_conf0 + best_idx * pred_conf1
    object_loss = sse(obj * best_conf, obj * best_giou.detach())

    # 6. No-Object Confidence Loss (Both boxes are penalized)
    no_object_loss = sse(noobj * pred_conf0, noobj * obj) + \
        sse(noobj * pred_conf1, noobj * obj)

    # 7. Classification Loss
    class_loss = sse(obj * prediction[..., :20], obj * target[..., :20])

    # 8. Total Loss
    total_loss = (
        lambda_coord * bbox_loss
        + object_loss
        + lambda_noobj * no_object_loss
        + class_loss
    )

    return total_loss
