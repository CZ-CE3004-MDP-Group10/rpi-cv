import sys
import cv2
import os
import socket
import time
import numpy as np
import tensorflow as tf
from PIL import Image
from object_detection.builders import model_builder
from object_detection.utils import label_map_util, config_util
from object_detection.utils import visualization_utils as viz_utils
from multiprocessing import Process, Queue
from combine import collate_output_image

class Detection:
    def __init__(self):
        # Path variables
        self.MODEL_DATA_PATH = "./model"
        self.CONFIG_PATH = self.MODEL_DATA_PATH + "/pipeline.config"
        self.category_index = label_map_util.create_category_index_from_labelmap(
            self.MODEL_DATA_PATH + '/label_map.pbtxt')
        self.RAW_IMAGES_PATH = "./images"
        self.INFERED_IMAGES_PATH = self.RAW_IMAGES_PATH + "/detected_images"

        # disable GPU/cuda
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        self.MIN_THRESHOLD = .50
        self.MAX_BOUNDING_BOXES = 100
        read_queue = Queue()
        write_queue = Queue()
        connected = False
        

        # start inference process
        p = Process(target=self.run_inference_for_single_image, args=(read_queue, write_queue))
        p.start()
        while True:
            if not connected:
                try:
                    # queue.put("1(4,1)")
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    print("Attempting to connect to RPI server...")
                    sock.connect(("192.168.10.1", 50001))
                    print(f"connected to {sock.getpeername()}")
                    connected = True
                except:
                    pass
            else:
                try:
                    with sock:
                        if not write_queue.empty():
                            sock.send(write_queue.get().encode('utf-8'))
                            pass
                        else:
                            message = sock.recv(1024).decode('utf8')
                            print(message)
                            if message == "stop":
                                collate_output_image()
                            print(f'file_name {message} (FILE RECEIVE START)')
                            file = open(f'{self.RAW_IMAGES_PATH}/{message}.jpg','wb')
                            l = sock.recv(1024)
                            while l:
                                file.write(l)
                                l = sock.recv(1024)
                            file.close()
                            print(f'file_name {message} (FILE RECEIVE SUCCESS)')
                            read_queue.put(message)
                except KeyboardInterrupt:
                    sock.close()
                    connected = False
                except socket.error as e:
                    print(e)
                    # sock.close()
                    # connected = False
                except Exception as e:
                    exception_type, exception_traceback = sys.exc_info()
                    line_number = exception_traceback.tb_lineno
                    print("Exception type: ", exception_type)
                    print("Line number: ", line_number)
                    print(f'ImageCV:{e}')

    @tf.function
    def detect_fn(self, image):
        """
        Detect objects in image.

        Args:
        image: (tf.tensor): 4D input image

        Returs:
        detections (dict): predictions that model made
        """
        start = time.time()
        print(f"(detect_fn) (START) predicting on {image}")
        image, shapes = self.detection_model.preprocess(image)
        prediction_dict = self.detection_model.predict(image, shapes)
        detections = self.detection_model.postprocess(prediction_dict, shapes)
        print(f"(detect_fn) (FINISH) {image} {time.time()-start:.2f}seconds ")
        return detections

    def run_inference_for_single_image(self , queue, write_queue):
        # model initialisation
        configs = config_util.get_configs_from_pipeline_file(self.CONFIG_PATH)  # import model config
        self.detection_model = model_builder.build(
            model_config=configs['model'], is_training=False)  # import model
        ckpt = tf.compat.v2.train.Checkpoint(model=self.detection_model)  # load model checkpoint
        ckpt.restore(os.path.join(f'{self.MODEL_DATA_PATH}/', 'ckpt-23')).expect_partial()
        print(f"MODEL LOADING COMPLETE")

        while True:
            if not queue.empty():
                image_name = queue.get()
                print(f"RUNNING INFERENCE ON...{image_name}")

                # convert image file into a numpy array
                image_np = np.array(Image.open(
                    f"{self.RAW_IMAGES_PATH}/{image_name}.jpg"))

                input_tensor = tf.convert_to_tensor(
                    np.expand_dims(image_np, 0), dtype=tf.float32)
                print(f"FILE CONVERSION ON... {image_name} complete")

                detections = self.detect_fn(input_tensor)
                # checking how many detections we got
                num_detections = int(detections.pop('num_detections'))
                # filtering out detections in order to get only the one that are indeed detections
                detections = {key: value[0, :num_detections].numpy()
                            for key, value in detections.items()}
                detections['num_detections'] = num_detections
                detections['detection_classes'] = detections['detection_classes'].astype(
                    np.int64)

                index = np.argmax(detections['detection_scores'])
                identified_class = detections['detection_classes'][index] + 1
                print(f"{identified_class} {detections['detection_scores'][index]}")

                if identified_class and detections['detection_scores'][index] > self.MIN_THRESHOLD:
                    message = f"AND|OBS{image_name}[{identified_class}]"
                    print(f"(CLASS IDENTIFIED) send message to Android and plot infered image")
                    print(f"(SEND) {message}")
                    write_queue.put(message)
                    viz_utils.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        detections['detection_boxes'],
                        detections['detection_classes'] + 1,
                        detections['detection_scores'],
                        self.category_index,
                        use_normalized_coordinates=True,
                        max_boxes_to_draw=self.MAX_BOUNDING_BOXES,
                        min_score_thresh=self.MIN_THRESHOLD,
                        line_thickness=5,
                        agnostic_mode=False)
                    cv2.imwrite(
                        f"{self.INFERED_IMAGES_PATH}/{image_name}.jpg", image_np)
                print(f"{image_name} VISUALISATION COMPLETE")


if __name__ == "__main__":
    main = Detection()
    print("Detection end")
