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

# Path variables
MODEL_DATA_PATH = "./model"
CONFIG_PATH = MODEL_DATA_PATH + "/pipeline.config"
CHECKPOINT_PATH = MODEL_DATA_PATH + "/"
category_index = label_map_util.create_category_index_from_labelmap(MODEL_DATA_PATH + '/label_map.pbtxt')
RAW_IMAGES_PATH = "./images"
INFERED_IMAGES_PATH = "./images/infered"

# Variables
HOST = "192.168.10.1"
PORT = 50001
MIN_THRESHOLD = .80
MAX_BOUNDING_BOXES = 100
MIN_X = 0.4
MAX_X = 0.6
MIN_Y = 0.6
MAX_Y = 0.8


def convert_image(received_image):
    read_time = time.time()
    frame = cv2.imread(received_image)
    image_np = np.array(frame)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    print("--- %s seconds <time taken to read image>) ---" % (time.time() - read_time))
    return image_np, input_tensor


@tf.function
def detect_fn(image):
    s = time.time()
    print("--- (detect_fn -> start of detect_fn) ---")
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    print("--- %s seconds (detect_fn -> end of detect_fn) ---" % (time.time() - s))
    return detections


def process_detections(input_tensor):
    detect_fn_time = time.time()
    detections = detect_fn(input_tensor)
    print("--- %s seconds <time taken for detect_fn function>) ---" % (time.time() - detect_fn_time))

    detection_pro_time = time.time()
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}

    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    print("--- %s seconds <time to display processed detection image> ---" % (time.time() - detection_pro_time))
    return detections


def get_center_of_label(coordinates):
    y_min, x_min, y_max, x_max = coordinates
    mid_x = (x_min + x_max) / 2
    mid_y = (y_min + y_max) / 2
    return mid_x, mid_y


def detection_information(detections):
    individual_detection = []
    classes = detections['detection_classes']
    scores = detections['detection_scores']
    boxes = detections['detection_boxes']

    for i in range(min(MAX_BOUNDING_BOXES, boxes.shape[0])):
        if scores is None or scores[i] > MIN_THRESHOLD:
            if classes[i] + 1 in category_index.keys():
                coordinates = tuple(boxes[i].tolist())  # [y_min, x_min, y_max, x_max]
                mid_x, mid_y = get_center_of_label(coordinates)
                # Check coordinates location
                if MIN_X <= mid_x <= MAX_X and MIN_Y <= mid_y <= MAX_Y:
                    class_name = category_index[classes[i] + 1]["name"]
                    label = [class_name, scores[i], mid_x, mid_y]
                    print(label)
                    individual_detection.append(label)
    return individual_detection


# Save image with detection in a folder
def show_inference(image_np_with_detections, detections):
    label_id_offset = 1
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'] + label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=MAX_BOUNDING_BOXES,
        min_score_thresh=MIN_THRESHOLD,
        line_thickness=1,
        agnostic_mode=False)
    write_image_time = time.time()
    cv2.imwrite(f"{INFERED_IMAGES_PATH}/processed_image_{image_count}.jpg", image_np_with_detections)
    print("--- %s seconds <time taken to write image to folder> ---" % (time.time() - write_image_time))


def run_inference_for_single_image(received_image):
    image_np, input_tensor = convert_image(received_image)
    detections = process_detections(input_tensor)
    # To get the class label details based on produced detections
    detected_items = detection_information(detections)
    # To get the image with bounding boxes
    if detected_items is not None:
        # sbuf.put_utf8("detected label" + str(detected_items[0]))
        show_inference(image_np, detections)
    print(f"{received_image} INFERENCED COMPLETED")


# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)
# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-23')).expect_partial()

# client socket connection to rpi server
try:
    # connect to server
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))
    print(f"connected to {sock.getpeername()}")
    with sock:
        sbuf = buffer.Buffer(sock)
        # once socket connected send command
        sbuf.put_utf8("CV|TAKIMG")
        print("cv takimg called")
        # client receive file name and size first
        file_name = sbuf.get_utf8()
        file_size = int(sbuf.get_utf8())
        print(f'{file_name} {file_size}')
        # client receive file data
        file_full_name = f'{RAW_IMAGES_PATH}/{file_name}.jpg'
        with open(file_full_name, 'wb') as f:
            remaining = file_size
            while remaining:
                chunk_size = 1024 if remaining >= 1024 else remaining
                chunk = sbuf.get_bytes(chunk_size)
                if not chunk: break
                f.write(chunk)
                remaining -= len(chunk)
            if remaining:
                print('File incomplete.  Missing', remaining, 'bytes.')
            else:
                print(f'File {file_name} received successfully.')
        # once file is receive, spawn new process to run image inference
        p = Process(target=run_inference_for_single_image, args=(file_full_name,)).start()
        # run_inference_for_single_image(file_name)
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


# # Run without Socket
# try:
#     # imgs = ["./images/test_estimate_set1_L_0.jpg",
#     #         "./images/test_estimate_set1_L_1.jpg", "./images/test_estimate_set1_R_0.jpg",
#     #         "./images/test_estimate_set1_R_1.jpg", "./images/test_estimate_set2_0.jpg"]
#     imgs2 = ["./images/30_set_1_005.jpg", "./images/30_set_5_050.jpg",
#              "./images/40_set_2_028.jpg", "./images/50_set_2_016.jpg"]
#
#     run_inference_for_single_image("./images/30_set_1_005.jpg")
#
#     # for i in range(len(imgs2)):
#     #     run_inference_for_single_image(imgs2[i])
#     #     image_count += 1
#
# except Exception as e:
#     print(f'ImageCV:{e}')
