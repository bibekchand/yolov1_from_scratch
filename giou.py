import torch


def giou(box1, box2, in_format):
    if in_format:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,
                                          0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,
                                          0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0] - box1[:, 2],
        box1[:, 1] - box1[:, 3], box1[:, 0] + box1[:, 2],
        box1[:, 1] + box1[:, 3]

    # Intersection + Convex hull box coordinates
    max_x1 = torch.max(b1_x1, b2_x1)
    min_x1 = torch.min(b1_x1, b2_x1)
    max_y1 = torch.max(b1_y1, b2_y1)
    min_y1 = torch.min(b1_y1, b2_y1)
    max_x2 = torch.max(b1_x2, b2_x2)
    min_x2 = torch.min(b1_x2, b2_x2)
    max_y2 = torch.max(b1_y2, b2_y2)
    min_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    intersection_area = (min_x2 - max_x1).clamp(0) * (min_y2 - max_y1).clamp(0)

    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - intersection_area
    iou = intersection_area / union_area

    # Convex Hull area
    convex_hull_area = (max_x2 - min_x1) * (max_y2 - min_y1)
    giou = iou - (convex_hull_area - union_area) / convex_hull_area
    return giou


box1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
box2 = torch.tensor([[20.0, 20.0, 30.0, 30.0]])
print(giou(box1, box2, True))
