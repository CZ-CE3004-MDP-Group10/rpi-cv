import sys
import cv2
import os
import socket
import time
import buffer
import numpy as np
import tensorflow as tf
from PIL import Image
from object_detection.builders import model_builder
from object_detection.utils import label_map_util, config_util
from object_detection.utils import visualization_utils as viz_utils
from multiprocessing import Process


class Detection:
    def __init__(self):
        # Path variables
        self.MODEL_DATA_PATH = "./model"
        self.CONFIG_PATH = self.MODEL_DATA_PATH + "/pipeline.config"
        self.CHECKPOINT_PATH = self.MODEL_DATA_PATH + "/"
        self.category_index = label_map_util.create_category_index_from_labelmap\
            (self.MODEL_DATA_PATH + '/label_map.pbtxt')
        self.RAW_IMAGES_PATH = "./images"
        self.INFERED_IMAGES_PATH = "./images/infered"

        self.MIN_THRESHOLD = .80
        self.MAX_BOUNDING_BOXES = 100

        # Load pipeline config and build a detection model
        configs = config_util.get_configs_from_pipeline_file(self.CONFIG_PATH)
        self.detection_model = model_builder.build(model_config=configs['model'], is_training=False)
        ckpt = tf.compat.v2.train.Checkpoint(model=self.detection_model)
        ckpt.restore(os.path.join(self.CHECKPOINT_PATH, 'ckpt-23')).expect_partial()

        try:
            #self.run_inference_for_single_image("30_set_1_005")
            # connect to server
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(("192.168.10.1", 50001))
            print(f"connected to {sock.getpeername()}")
            with sock:
                self.sbuf = buffer.Buffer(sock)
                file_name = self.sbuf.get_utf8()
                file_size = int(self.sbuf.get_utf8())
                print(f'{file_name} {file_size}')

                with open(f'{self.RAW_IMAGES_PATH}/{file_name}.jpg', 'wb') as f:
                    remaining = file_size
                    while remaining:
                        chunk_size = 1024 if remaining >= 1024 else remaining
                        chunk = self.sbuf.get_bytes(chunk_size)
                        if not chunk: break
                        f.write(chunk)
                        remaining -= len(chunk)
                    if remaining:
                        print('File incomplete.  Missing', remaining, 'bytes.')
                    else:
                        print(f'File {file_name} received successfully.')

                print("RUNNING Process")
                # p = Process(target=self.run_inference_for_single_image, args=(file_name,)).start()
                self.run_inference_for_single_image(file_name)
        except KeyboardInterrupt:
            sock.close()
        except socket.timeout as e:
            print(e)
            sock.close()
        except Exception as e:
            exception_type, exception_object, exception_traceback = sys.exc_info()
            line_number = exception_traceback.tb_lineno
            print("Exception type: ", exception_type)
            print("Line number: ", line_number)
            print(f'ImageCV:{e}')

    @tf.function
    def detect_fn(self, image):
        s = time.time()
        print("--- (detect_fn -> start of detect_fn) ---")
        image, shapes = self.detection_model.preprocess(image)
        prediction_dict = self.detection_model.predict(image, shapes)
        detections = self.detection_model.postprocess(prediction_dict, shapes)
        print("--- %s seconds (detect_fn -> end of detect_fn) ---" % (time.time() - s))
        return detections

    def run_inference_for_single_image(self, image_name):
        print(image_name)
        print("CONVERT IMAGE")
        print(f"{self.RAW_IMAGES_PATH}/{image_name}.jpg")
        frame = cv2.imread(f"{self.RAW_IMAGES_PATH}/{image_name}.jpg")
        image_np = np.array(frame)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

        print("PROCESS DETECTION")
        detections = self.detect_fn(input_tensor)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        index = np.argmax(detections['detection_scores'])
        identified_class = detections['detection_classes'][index] + 1
        print(identified_class)

        if identified_class:
            self.sbuf.put_utf8(f"AND|OBS{image_name}[{identified_class}]")
            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np,
                detections['detection_boxes'],
                detections['detection_classes'] + 1,
                detections['detection_scores'],
                self.category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=self.MAX_BOUNDING_BOXES,
                min_score_thresh=self.MIN_THRESHOLD,
                line_thickness=1,
                agnostic_mode=False)

            cv2.imwrite(f"{self.INFERED_IMAGES_PATH}/{image_name}.jpg", image_np)
        print(f"{image_name} INFERENCED COMPLETED")


if __name__ == "__main__":
    main = Detection()
    while True:
        pass
    # def get_center_of_label(self, coordinates):
    #     y_min, x_min, y_max, x_max = coordinates
    #     mid_x = (x_min + x_max) / 2
    #     mid_y = (y_min + y_max) / 2
    #     return mid_x, mid_y

    # def detection_information(self, detections):
    #     individual_detection = []
    #     classes = detections['detection_classes']
    #     scores = detections['detection_scores']
    #     boxes = detections['detection_boxes']
    #
    #     for i in range(min(self.MAX_BOUNDING_BOXES, boxes.shape[0])):
    #         if scores is None or scores[i] > self.MIN_THRESHOLD:
    #             if classes[i] + 1 in self.category_index.keys():
    #                 coordinates = tuple(boxes[i].tolist())  # [y_min, x_min, y_max, x_max]
    #                 mid_x, mid_y = get_center_of_label(coordinates)
    #                 class_name = category_index[classes[i] + 1]["name"]
    #                 label = f"{class_name}: {scores[i]} x:{mid_x} y:{mid_y}"
    #                 print(label)
    #                 # Check coordinates location
    #                 if MIN_X <= mid_x <= MAX_X and MIN_Y <= mid_y <= MAX_Y:
    #                     individual_detection.append(class_name)
    #     return individual_detection

    # Save image with detection in a folder
    # def show_inference(self, image_np_with_detections, detections):
    #     label_id_offset = 1
    #     viz_utils.visualize_boxes_and_labels_on_image_array(
    #         image_np_with_detections,
    #         detections['detection_boxes'],
    #         detections['detection_classes'] + label_id_offset,
    #         detections['detection_scores'],
    #         self.category_index,
    #         use_normalized_coordinates=True,
    #         max_boxes_to_draw=self.MAX_BOUNDING_BOXES,
    #         min_score_thresh=self.MIN_THRESHOLD,
    #         line_thickness=1,
    #         agnostic_mode=False)
    #     write_image_time = time.time()
    #     cv2.imwrite(f"{self.INFERED_IMAGES_PATH}/{file_name}", image_np_with_detections)
    #     print("--- %s seconds <time taken to write image to folder> ---" % (time.time() - write_image_time))
