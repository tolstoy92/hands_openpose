import numpy as np
from object_detection.utils import visualization_utils as vis_util


class HandsDetector:

    @staticmethod
    def detect_objects(image_np, sess, detection_graph, category_index):
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        # Visualization of the results of a detection.

        (class_name, box, boxes_lst) = vis_util.visualize_boxes_and_labels_on_image_array(
            image_np, np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores), category_index,
            use_normalized_coordinates=True,
            line_thickness=4, max_boxes_to_draw=2)

        return image_np, class_name, box, boxes_lst

    @staticmethod
    def searching_area_by_cnts(boxes_lst, width, height):
        cnts = []
        for box in boxes_lst:
            y_min, x_min, y_max, x_max = box[0], box[1], box[2], box[3]
            #cnts.append((int(x_max * width), int(y_max * height)))
            #cnts.append((int(x_min * width), int(y_min * height)))

            x = (int(x_max * width) - (int(x_min * width)))//2 + (int(x_min * width))
            y = (int(y_max * height) - (int(y_min * height)))//2 + (int(y_min * height))

            cnts.append((x, y))
            #cnt = (int((y_min + (y_max - y_min)//2) * height), int((x_min + (x_max - x_min)//2) * width))
            #cnts.append(cnt)
        return cnts
