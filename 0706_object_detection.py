import tensorflow as tf

def bbox_giou(boxes1, boxes2):
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:]*0.5,
                        boxes1[..., :2] + boxes1[..., 2]:*0.5], axis = -1)

    boxes2 = tf.concat([boxes2[..., 2] - boxes2[..., 2:]*0.5
                        boxes2[..., 2] + boxes2[..., 2:]*0.5], axis = -1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, , 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area/union_area

    