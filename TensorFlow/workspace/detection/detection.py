import os, cv2, time, socket
from random import randint
import numpy as np
from PIL import Image
import tensorflow as tf
from multiprocessing import Process, Queue
from object_detection.utils import label_map_util, config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from matplotlib import pyplot as plt

MODEL_DATA_PATH = "./model"
CONFIG_PATH = MODEL_DATA_PATH + "/pipeline.config"
CHECKPOINT_PATH = MODEL_DATA_PATH + "/"
category_index = label_map_util.create_category_index_from_labelmap(MODEL_DATA_PATH + '/label_map.pbtxt')
IMAGES_PATH = "./img1"
# image_detect_q = Queue()

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)
# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-23')).expect_partial()


@tf.function
def detect_fn(image):
    s = time.time()
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    print("--- %s seconds (detecting) ---" % (time.time() - s))
    return detections


def read_image(received_data):
    print("running read_image ....")
    count = randint(1, 1000)
    # for image in os.listdir(IMAGES_PATH):
    # count = count+1
    # image_np = load_image_into_numpy_array(IMAGES_PATH + '/' + image)

    # image_np = np.tile(np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8) #BW
    read_time = time.time()
    frame = cv2.imread(received_data)
    image_np = np.array(frame)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    print("--- %s seconds (end of reading image) ---" % (time.time() - read_time))
    detections = detect_fn(input_tensor)

    d_time = time.time()

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}

    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'] + label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=100,
        min_score_thresh=.50,
        line_thickness=1,
        agnostic_mode=False)

    print("--- %s seconds (end of detection methods ...) ---" % (time.time() - d_time))
    w_image = time.time()
    cv2.imwrite("./output_images/o" + str(count) + ".jpg", image_np_with_detections)
    print("--- %s seconds (end of writting image...) ---" % (time.time() - w_image))
    print("output image")

# start_time = time.time()
# print("--- %s seconds (load model) ---" % (time.time() - start_time))

def send_message(message):
    sock.send(message.encode("utf-8"))


def receive_file_info():
    received = sock.recv(1024).decode('utf-8')
    print("received: " + received)
    filename, filesize = received.split('|')


def write_file():
    with open(file_name, 'wb') as f:
        while True:
            bytes_read = sock.recv(1024)
            if not bytes_read:
                break
            f.write(bytes_read)
        print("File written")


try:
    print("Attempt connection")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("192.168.10.1", 50001))
    print(f"connected to {sock.getpeername()}")

    send_message("CV|TAKIMG")
    file_name = "received_data_1.jpg"
    receive_file_info()
    write_file()

    read_image(file_name)
    # new process to do detection
    # p = Process(target=read_image, args=[file_name], daemon=True)
    # p.start()

    print("socket closed")
except KeyboardInterrupt:
    sock.close()
except socket.timeout as e:
    print(e)
    sock.close()
except Exception as e:
    print(f'Imagecv:{e}')
