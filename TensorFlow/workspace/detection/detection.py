import os, cv2, time, socket
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
    count = 0
    # for image in os.listdir(IMAGES_PATH):
    # count = count+1
    # image_np = load_image_into_numpy_array(IMAGES_PATH + '/' + image)

    # image_np = np.tile(np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8) #BW
    frame = cv2.imread(received_data)
    image_np = np.array(frame)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor, detection_model)

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
        min_score_thresh=.75,
        line_thickness=1,
        agnostic_mode=False)

    count = count + 1

    cv2.imwrite("./output_images/output_image" + str(count) + ".jpg", image_np_with_detections)


# start_time = time.time()
# detection_model = load_model()
# print("--- %s seconds (load model) ---" % (time.time() - start_time))

# detect_time = time.time()
# read_image(detection_model)
# print("--- %s seconds (read image) ---" % (time.time() - detect_time))
# read_image(detection_model)

try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect("192.168.10.1", 50001)

    message = "CV|TAKIMG"
    message = message.encode("utf-8")
    sock.send(message)
    print(message)
    i = 0
    while True:
        # f = open(f'received_data_{i}.jpg', 'wb')
        # received_data = sock.recv(1024)
        # while received_data:
        #     print("reading image")
        #     sock.write(received_data)
        #     received_data.recv(1024)
        # received_data.close()
        #
        # i = i + 1
        received_name = f'received_data_{i}.jpg'
        with open(received_name, 'wb') as f:
            while True:
                data = sock.recv(1024)
                if not data:
                    break
                f.write(data)
        i = i + 1
        # new process to do detection
        p = Process(target=read_image, args=(received_name,), daemon=True)
        p.start()

except Exception as e:
    print(f'Imagecv:{e}')
