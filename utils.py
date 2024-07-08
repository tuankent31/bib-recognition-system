import csv
import shutil

import pymssql
from config import Config
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import pandas as pd
from ultralytics import YOLO
from vietocr.tool.predictor import Predictor
from easyocr import Reader
from PIL import Image
from scipy.interpolate import interp1d
from sort.sort import *

def load_bib_detector(filePath):
    bib_detector_model = YOLO(filePath)
    return bib_detector_model

def load_esrgan_model(filePath):
    esrgan_model = hub.load(filePath)
    return esrgan_model

def load_coco_model(filePath):
    coco_model = YOLO(filePath)
    return coco_model

def load_vietocr_engine():
    vietocr_engine = Predictor(Config.vietocr_config)
    return vietocr_engine

def load_easyocr_engine():
    easyocr_engine = Reader(['vi', 'en'], gpu=True)
    return easyocr_engine

def get_database_connection():
    return pymssql.connect(Config.SQL_SERVER, Config.SQL_USER, Config.SQL_PASSWORD, Config.SQL_DATABASE)

def preprocess_image(image):
    image = tf.convert_to_tensor(image, dtype=tf.uint8)
    if image.shape[-1] == 4:
        image = image[..., :3]
    image = tf.cast(image, tf.float32)
    return tf.expand_dims(image, 0)

def postprocess_image(image):
    image = tf.clip_by_value(image, 0, 255)
    image = tf.cast(image, tf.uint8)
    return image

def preprocess_image1(image):
    im_nr = cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 15, 15)

    return im_nr

def save_cropped_image(image, index, save_folder):
    filename = f"plate_region_{index}.jpg"
    filepath = os.path.join(save_folder, filename)
    cv2.imwrite(filepath, image)
    return filename

def score_frame(model, frame):
    results = model(frame)
    results = results[0].boxes
    labels = results.cls
    cord = results.xyxyn
    conf = results.conf
    return labels, cord, conf

def check_bib_number(plate_number):
    conn = get_database_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT Name, Time, Pace, DistanceType FROM dbo.Runners WHERE BIB = %s", (plate_number,))
    return cursor.fetchone()

def plot_boxes(detect_model, process_model, ocr_engine1, ocr_engine2, results, frame, save_folder):
    labels, cord, conf = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    annotated_frame = frame.copy()
    cropped_images = []

    for i in range(n):
        row = cord[i]
        detection_conf = conf[i].item()
        if detection_conf >= 0.2:
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2),
                          (0, 255, 0) if detect_model.names[int(labels[i])] != 'person' else (0, 0, 255), 2)
            plate_region = frame[y1:y2, x1:x2]

            cropped_images.append(plate_region)

    result_data = []

    conn = get_database_connection()
    cursor = conn.cursor()

    for j, hr_image_np in enumerate(cropped_images):
        row = cord[j]

        pil_image = Image.fromarray(hr_image_np)
        vietocr_result = ocr_engine1.predict(pil_image)
        cursor.execute("SELECT Name, Time, Pace, DistanceType FROM dbo.Runners WHERE BIB = %s", (vietocr_result,))
        result = cursor.fetchone()
        if result:
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                row[3] * y_shape)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, vietocr_result, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                        (255, 255, 255), 5)
            result_data.append({
                "bib": vietocr_result,
                "name": result[0],
                "time": result[1],
                "pace": result[2],
                "distance_type": result[3],
                "image_filename": save_cropped_image(hr_image_np, j, save_folder)
            })
        else:
            hr_image = preprocess_image(hr_image_np)
            fake_image = process_model(hr_image)
            fake_image = postprocess_image(tf.squeeze(fake_image))
            proc_image = preprocess_image1(fake_image.numpy())

            vietocr_result_sr = ocr_engine1.predict(Image.fromarray(proc_image))
            cursor.execute("SELECT Name, Time, Pace, DistanceType FROM dbo.Runners WHERE BIB = %s", (vietocr_result_sr,))
            result = cursor.fetchone()
            if result:
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                    row[3] * y_shape)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, vietocr_result_sr, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                            (255, 255, 255), 5)
                result_data.append({
                    "bib": vietocr_result_sr,
                    "name": result[0],
                    "time": result[1],
                    "pace": result[2],
                    "distance_type": result[3],
                    "image_filename": save_cropped_image(hr_image_np, j, save_folder)
                })
            else:
                easyocr_result = ocr_engine2.readtext(hr_image_np)

                if easyocr_result:
                    plate_number= easyocr_result[0][1]
                    cursor.execute("SELECT Name, Time, Pace, DistanceType FROM dbo.Runners WHERE BIB = %s",
                                   (plate_number,))
                    result = cursor.fetchone()
                    if result:
                        x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                            row[3] * y_shape)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, plate_number, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                    (255, 255, 255), 5)
                        result_data.append({
                            "bib": plate_number,
                            "name": result[0],
                            "time": result[1],
                            "pace": result[2],
                            "distance_type": result[3],
                            "image_filename": save_cropped_image(hr_image_np, j, save_folder)
                        })
                else:
                    easyocr_result_sr = ocr_engine2.readtext(proc_image)
                    if easyocr_result_sr:
                        plate_number = easyocr_result_sr[0][1]
                        cursor.execute("SELECT Name, Time, Pace, DistanceType FROM dbo.Runners WHERE BIB = %s",
                                       (plate_number,))
                        result = cursor.fetchone()
                        if result:
                            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                                row[3] * y_shape)
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(annotated_frame, plate_number, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                        (255, 255, 255), 5)
                            result_data.append({
                                "bib": plate_number,
                                "name": result[0],
                                "time": result[1],
                                "pace": result[2],
                                "distance_type": result[3],
                                "image_filename": save_cropped_image(hr_image_np, j, save_folder)
                            })

    return annotated_frame, result_data

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def interpolate_bounding_boxes(data):
    frame_numbers = np.array([int(row['frame_nmr']) for row in data])
    pp_ids = np.array([int(float(row['people_id'])) for row in data])
    pp_bboxes = np.array([list(map(float, row['people_bbox'][1:-1].split())) for row in data])
    bib_plate_bboxes = np.array([list(map(float, row['bib_plate_bbox'][1:-1].split())) for row in data])

    interpolated_data = []
    unique_pp_ids = np.unique(pp_ids)
    for pp_id in unique_pp_ids:
        frame_numbers_ = [p['frame_nmr'] for p in data if int(float(p['people_id'])) == int(float(pp_id))]

        pp_mask = pp_ids == pp_id
        pp_frame_numbers = frame_numbers[pp_mask]
        pp_bboxes_interpolated = []
        bib_plate_bboxes_interpolated = []

        first_frame_number = pp_frame_numbers[0]

        for i in range(len(pp_bboxes[pp_mask])):
            frame_number = pp_frame_numbers[i]
            car_bbox = pp_bboxes[pp_mask][i]
            license_plate_bbox = bib_plate_bboxes[pp_mask][i]

            if i > 0:
                prev_frame_number = pp_frame_numbers[i-1]
                prev_car_bbox = pp_bboxes_interpolated[-1]
                prev_license_plate_bbox = bib_plate_bboxes_interpolated[-1]

                if frame_number - prev_frame_number > 1:
                    frames_gap = frame_number - prev_frame_number
                    x = np.array([prev_frame_number, frame_number])
                    x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)
                    interp_func = interp1d(x, np.vstack((prev_car_bbox, car_bbox)), axis=0, kind='linear')
                    interpolated_car_bboxes = interp_func(x_new)
                    interp_func = interp1d(x, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0, kind='linear')
                    interpolated_license_plate_bboxes = interp_func(x_new)

                    pp_bboxes_interpolated.extend(interpolated_car_bboxes[1:])
                    bib_plate_bboxes_interpolated.extend(interpolated_license_plate_bboxes[1:])

            pp_bboxes_interpolated.append(car_bbox)
            bib_plate_bboxes_interpolated.append(license_plate_bbox)

        for i in range(len(pp_bboxes_interpolated)):
            frame_number = first_frame_number + i
            row = {}
            row['frame_nmr'] = str(frame_number)
            row['people_id'] = str(pp_id)
            row['people_bbox'] = ' '.join(map(str, pp_bboxes_interpolated[i]))
            row['bib_plate_bbox'] = ' '.join(map(str, bib_plate_bboxes_interpolated[i]))

            if str(frame_number) not in frame_numbers_:
                row['bib_plate_bbox_score'] = '0'
                row['bib_number'] = '0'
                row['bib_number_score'] = '0'
            else:
                original_row = [p for p in data if int(p['frame_nmr']) == frame_number and int(float(p['people_id'])) == int(float(pp_id))][0]
                row['bib_plate_bbox_score'] = original_row['bib_plate_bbox_score'] if 'bib_plate_bbox_score' in original_row else '0'
                row['bib_number'] = original_row['bib_number'] if 'bib_number' in original_row else '0'
                row['bib_number_score'] = original_row['bib_number_score'] if 'bib_number_score' in original_row else '0'

            interpolated_data.append(row)

    return interpolated_data

def process_video(detect_model, coco_model, ocr_engine, video_path, upload_folder):
    results = {}
    mot_tracker = Sort()

    coco_model = coco_model
    bib_detector = detect_model

    cap = cv2.VideoCapture(video_path)

    peoples = 0
    frame_nmr = -1
    ret = True
    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        if ret:
            results[frame_nmr] = {}

            detections = coco_model(frame)[0]
            detections_ = []
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) == peoples:
                    detections_.append([x1, y1, x2, y2, score])

            track_ids = mot_tracker.update(np.asarray(detections_))

            bib_plates = bib_detector(frame)[0]
            for bib_plate in bib_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = bib_plate
                xpp1, ypp1, xpp2, ypp2, pp_id = get_people(bib_plate, track_ids)

                if pp_id != -1:
                    bib_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                    bib_plate_text, bib_plate_text_score = read_bib_plate(ocr_engine, bib_plate_crop)

                    if bib_plate_text is not None:
                        results[frame_nmr][pp_id] = {'people': {'bbox': [xpp1, ypp1, xpp2, ypp2]},
                                                     'bib_plate': {'bbox': [x1, y1, x2, y2],
                                                                   'text': bib_plate_text,
                                                                   'bbox_score': score,
                                                                   'text_score': bib_plate_text_score}}
    write_csv(results, os.path.join(upload_folder, 'results.csv'))

def interpolate_results(csv_path, upload_folder):
    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file)
        data = list(reader)

    interpolated_data = interpolate_bounding_boxes(data)

    header = ['frame_nmr', 'people_id', 'people_bbox', 'bib_plate_bbox', 'bib_plate_bbox_score', 'bib_number', 'bib_number_score']
    with open(os.path.join(upload_folder, 'results_interpolated.csv'), 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        writer.writerows(interpolated_data)


def get_people(bib_plate, people_track_ids):
    global pp_index
    x1, y1, x2, y2, score, class_id = bib_plate

    foundIt = False
    for j in range(len(people_track_ids)):
        xpp1, ypp1, xpp2, ypp2, pp_id = people_track_ids[j]

        if x1 > xpp1 and y1 > ypp1 and x2 < xpp2 and y2 < ypp2:
            pp_index = j
            foundIt = True
            break

    if foundIt:
        return people_track_ids[pp_index]

    return -1, -1, -1, -1, -1

def read_bib_plate(ocr_engine, bib_plate_crop):
    detections = ocr_engine.readtext(bib_plate_crop)

    for detection in detections:
        bbox, test, score = detection


        return test, score

    return 0, 0

def write_csv(results, output_path):
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'people_id', 'people_bbox',
                                                'bib_plate_bbox', 'bib_plate_bbox_score', 'bib_number',
                                                'bib_number_score'))

        for frame_nmr in results.keys():
            for pp_id in results[frame_nmr].keys():
                print(results[frame_nmr][pp_id])
                if 'people' in results[frame_nmr][pp_id].keys() and \
                   'bib_plate' in results[frame_nmr][pp_id].keys() and \
                   'text' in results[frame_nmr][pp_id]['bib_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            pp_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][pp_id]['people']['bbox'][0],
                                                                results[frame_nmr][pp_id]['people']['bbox'][1],
                                                                results[frame_nmr][pp_id]['people']['bbox'][2],
                                                                results[frame_nmr][pp_id]['people']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][pp_id]['bib_plate']['bbox'][0],
                                                                results[frame_nmr][pp_id]['bib_plate']['bbox'][1],
                                                                results[frame_nmr][pp_id]['bib_plate']['bbox'][2],
                                                                results[frame_nmr][pp_id]['bib_plate']['bbox'][3]),
                                                            results[frame_nmr][pp_id]['bib_plate']['bbox_score'],
                                                            results[frame_nmr][pp_id]['bib_plate']['text'],
                                                            results[frame_nmr][pp_id]['bib_plate']['text_score'])
                            )
        f.close()


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=2, line_length_x=20, line_length_y=20):
    x1, y1 = top_left
    x2, y2 = bottom_right
    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)
    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)
    return img

def update_database_with_images(bib_number, filenames):
    conn = get_database_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT Images FROM dbo.Runners WHERE BIB = %s", (bib_number,))
    result = cursor.fetchone()

    existing_images = []
    if result and result[0]:
        existing_images = result[0].split()

    updated_images = existing_images + filenames
    updated_images_str = " ".join(updated_images)

    cursor.execute("UPDATE dbo.Runners SET Images = %s WHERE BIB = %s", (updated_images_str, bib_number))
    conn.commit()

def update_database_with_video_name(csv_file_path, video_name):
    conn = get_database_connection()
    cursor = conn.cursor()

    results = pd.read_csv(csv_file_path)

    for bib_number in results['bib_number']:
        cursor.execute("SELECT Notes FROM dbo.Runners WHERE BIB = %s", bib_number)
        row = cursor.fetchone()

        if row:
            current_notes = row[0] if row[0] else ''
            new_notes = f"{current_notes} {video_name}".strip()
            cursor.execute("UPDATE dbo.Runners SET Notes = %s WHERE BIB = %s", (new_notes, bib_number))

    conn.commit()
    conn.close()


# Clean up the uploads folder on exit
def cleanup_uploads_folder(file):
    shutil.rmtree(file)
    os.makedirs(file)

