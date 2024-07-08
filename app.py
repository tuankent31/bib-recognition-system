from flask import Flask, request, render_template, send_from_directory, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from werkzeug.security import check_password_hash, generate_password_hash

from utils import *
from config import Config

import ast
import pandas as pd
import numpy as np

import atexit
import tensorflow as tf


app = Flask(__name__)
app.secret_key = 'hehehehe'

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

upload_folder = Config.UPLOAD_FOLDER
os.makedirs(upload_folder, exist_ok=True)

img_bucket = 'static/images'




upload_folder = Config.UPLOAD_FOLDER

bib_detector_model = load_bib_detector(Config.BIB_DETECTION_MODEL_PATH)

coco_model = load_coco_model(Config.COCO_MODEL_PATH)

esrgan_model = load_esrgan_model(Config.ESRGAN_MODEL_PATH)

vietocr_engine = load_vietocr_engine()

easyocr_reader = load_easyocr_engine()

users = {
    'admin': generate_password_hash('password123')
}

class User(UserMixin):
    def __init__(self, id):
        self.id = id

# Database connection details
server = 'DESKTOP-S80S0CJ\\SQLEXPRESS'
user = 'sa'
password = '1234567890'
database = 'bibdata'

@login_manager.user_loader
def load_user(user_id):
    return User(user_id) if user_id in users else None

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and check_password_hash(users[username], password):
            user = User(username)
            login_user(user)
            flash('Logged in successfully.')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password.')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully.')
    return redirect(url_for('login'))


@app.route('/index')
@login_required
def index():
    conn = get_database_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT Gender, COUNT(*) as count FROM dbo.Runners GROUP BY Gender")
    gender_data = cursor.fetchall()

    gender_ratio = {
        'Male': 0,
        'Female': 0
    }

    for row in gender_data:
        gender_ratio[row[0]] = row[1]

    print(gender_ratio)

    return render_template('index.html', gender_ratio=gender_ratio)


@app.route('/bib-recognition/photo', methods=['GET', 'POST'])
@login_required
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(upload_folder, file.filename)
            file.save(filepath)
            image = Image.open(filepath)
            frame = np.array(image)

            results = score_frame(bib_detector_model, frame)
            annotated_frame, result_data = plot_boxes(bib_detector_model, esrgan_model, vietocr_engine, easyocr_reader,results, frame, upload_folder)

            # Save the annotated image
            annotated_image_path = os.path.join(upload_folder, 'annotated_' + file.filename)
            Image.fromarray(annotated_frame).save(annotated_image_path)

            return render_template('result.html', filename='annotated_' + file.filename, results=result_data)

    return render_template('bib_recog_photo.html')


@app.route('/bib-recognition/video', methods=['GET', 'POST'])
def video_recog():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(os.path.join(upload_folder, filename))
            process_video(bib_detector_model, coco_model, easyocr_reader,os.path.join(upload_folder, filename), upload_folder)

            interpolate_results(os.path.join(upload_folder, 'results.csv'), upload_folder)

            # Call the function to update the database
            update_database_with_video_name(os.path.join(upload_folder, 'results_interpolated.csv'), filename)

            session['video_filename'] = filename

            return redirect(url_for('show_csv'))
    return render_template('bib_recog_video.html')



@app.route('/show-csv')
def show_csv():
    csv_file_path = os.path.join(upload_folder, 'results_interpolated.csv')
    data = []
    with open(csv_file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return render_template('show_csv.html', data=data)


@app.route('/show_csv')
def show_csv_pandas():
    data = pd.read_csv(os.path.join(upload_folder, 'results_interpolated.csv')).to_dict(orient='records')
    return render_template('show_csv.html', data=data)


@app.route('/visualize')
def visualize():
    filename = session.get('video_filename')
    if not filename:
        return redirect(url_for('video_recog'))

    video_path = os.path.join(upload_folder, filename)
    results = pd.read_csv(os.path.join(upload_folder, 'results_interpolated.csv'))

    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(os.path.join(upload_folder, 'out.mp4'), fourcc, fps, (width, height))

    bib_plate = {}
    for pp_id in np.unique(results['people_id']):
        max_ = np.amax(results[results['people_id'] == pp_id]['bib_number_score'])
        bib_plate[pp_id] = {'bib_crop': None, 'bib_plate_number':
            results[(results['people_id'] == pp_id) & (results['bib_number_score'] == max_)]['bib_number'].iloc[0]}
        cap.set(cv2.CAP_PROP_POS_FRAMES,
                results[(results['people_id'] == pp_id) & (results['bib_number_score'] == max_)]['frame_nmr'].iloc[0])
        ret, frame = cap.read()

        bbox_str = \
        results[(results['people_id'] == pp_id) & (results['bib_number_score'] == max_)]['bib_plate_bbox'].iloc[0]
        bbox_str = bbox_str.replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ',')
        x1, y1, x2, y2 = ast.literal_eval(bbox_str)

        bib_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
        bib_crop = cv2.resize(bib_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))
        bib_plate[pp_id]['bib_crop'] = bib_crop

    frame_nmr = -1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret = True
    while ret:
        ret, frame = cap.read()
        frame_nmr += 1
        if ret:
            df_ = results[results['frame_nmr'] == frame_nmr]
            for row_indx in range(len(df_)):
                pp_x1, pp_y1, pp_x2, pp_y2 = ast.literal_eval(
                    df_.iloc[row_indx]['people_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(
                        ' ', ','))
                draw_border(frame, (int(pp_x1), int(pp_y1)), (int(pp_x2), int(pp_y2)), (0, 255, 0), 3, line_length_x=20,
                            line_length_y=20)

                bbox_str = df_.iloc[row_indx]['bib_plate_bbox']
                bbox_str = bbox_str.replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ',')
                x1, y1, x2, y2 = ast.literal_eval(bbox_str)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

                bib_crop = bib_plate[df_.iloc[row_indx]['people_id']]['bib_crop']
                bib_crop = cv2.resize(bib_crop, (240, 100), interpolation=cv2.INTER_LINEAR)
                H, W, _ = bib_crop.shape

                try:
                    frame[int(pp_y1) - H:int(pp_y1), int((pp_x2 + pp_x1 - W) / 2):int((pp_x2 + pp_x1 + W) / 2),
                    :] = bib_crop
                    frame[int(pp_y1) - H:int(pp_y1) - H, int((pp_x2 + pp_x1 - W) / 2):int((pp_x2 + pp_x1 + W) / 2),
                    :] = (255, 255, 255)

                    (text_width, text_height), _ = cv2.getTextSize(
                        bib_plate[df_.iloc[row_indx]['people_id']]['bib_plate_number'], cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                        3)
                    cv2.putText(frame, bib_plate[df_.iloc[row_indx]['people_id']]['bib_plate_number'],
                                (int((pp_x2 + pp_x1 - text_width) / 2), int(pp_y1 - H - 60 + (text_height / 2))),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
                except:
                    pass
            out.write(frame)

    out.release()
    cap.release()

    return redirect(url_for('show_video'))


@app.route('/show-video')
def show_video():
    return render_template('show_video.html', video_url=url_for('static', filename='uploads/out.mp4'))


@app.route('/batch-image-recognition', methods=['GET', 'POST'])
@login_required
def batch_image_recognition():
    if request.method == 'POST':
        recognized_bibs = {}

        for filename in os.listdir(img_bucket):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(img_bucket, filename)


                image = Image.open(filepath)
                frame = np.array(image)

                results = score_frame(bib_detector_model, frame)
                labels, cord, conf = results

                for i in range(len(labels)):
                    row = cord[i]
                    detection_conf = conf[i].item()
                    if detection_conf >= 0.2:
                        x1, y1, x2, y2 = int(row[0] * frame.shape[1]), int(row[1] * frame.shape[0]), int(
                            row[2] * frame.shape[1]), int(row[3] * frame.shape[0])
                        plate_region = frame[y1:y2, x1:x2]


                        # VietOCR normal plate_region
                        pil_image = Image.fromarray(plate_region)
                        plate_number = vietocr_engine.predict(pil_image)
                        result = check_bib_number(plate_number)

                        if result:
                            if plate_number not in recognized_bibs:
                                recognized_bibs[plate_number] = []
                            recognized_bibs[plate_number].append(filename)
                        else:
                            hr_image = preprocess_image(plate_region)
                            fake_image = esrgan_model(hr_image)
                            fake_image = postprocess_image(tf.squeeze(fake_image))
                            proc_img = preprocess_image1(fake_image.numpy())

                            # VietOCR with super-resolution plate_region
                            pil_image_sr = Image.fromarray(proc_img)
                            plate_number = vietocr_engine.predict(pil_image_sr)
                            result = check_bib_number(plate_number)
                            if result:
                                if plate_number not in recognized_bibs:
                                    recognized_bibs[plate_number] = []
                                recognized_bibs[plate_number].append(filename)
                            else:
                                # EasyOCR normal plate_region
                                easyocr_result = easyocr_reader.readtext(plate_region)

                                if easyocr_result:
                                    plate_number, confidence = easyocr_result[0][1], easyocr_result[0][2]
                                    if (confidence > 0.8):
                                        result = check_bib_number(plate_number)
                                        if result:
                                            if plate_number not in recognized_bibs:
                                                recognized_bibs[plate_number] = []
                                            recognized_bibs[plate_number].append(filename)
                                    else:
                                        # EasyOCR super-res plate_region
                                        easyocr_result_sr = easyocr_reader.readtext(proc_img, detail=0)
                                        if easyocr_result_sr:
                                            plate_number = easyocr_result_sr[0]
                                            result = check_bib_number(plate_number)
                                            if result:
                                                if plate_number not in recognized_bibs:
                                                    recognized_bibs[plate_number] = []
                                                recognized_bibs[plate_number].append(filename)
                                            else:
                                                recognized_bibs[plate_number] = []


        for bib_number, filenames in recognized_bibs.items():
            update_database_with_images(bib_number, filenames)
        print('Batch processing completed')

        return '', 200  # Return a successful response

    return render_template('batch_image_recognition.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(upload_folder, filename)

# Pagination settings
RESULTS_PER_PAGE = 10  # Number of results per page

@app.route('/bib_management')
@app.route('/bib_management/<int:page>')
@login_required
def bib_management(page=1):
    conn = get_database_connection()
    cursor = conn.cursor()

    search_query = {}
    for col in ['BIB', 'Name', 'Age', 'Gender', 'National', 'Time', 'Pace']:
        search_query[col] = request.args.get(col, '')

    base_query = "SELECT BIB, Name, Age, Gender, [National], Time, Pace FROM dbo.Runners WHERE 1=1"
    count_query = "SELECT COUNT(*) FROM dbo.Runners WHERE 1=1"

    for col, value in search_query.items():
        if value:
            base_query += f" AND {col} LIKE %s"
            count_query += f" AND {col} LIKE %s"

    base_query += " ORDER BY BIB OFFSET %s ROWS FETCH NEXT %s ROWS ONLY"

    cursor.execute(count_query, tuple(f"%{value}%" for value in search_query.values() if value))
    total_rows = cursor.fetchone()[0]
    total_pages = (total_rows + RESULTS_PER_PAGE - 1) // RESULTS_PER_PAGE  # Ceiling division

    offset = (page - 1) * RESULTS_PER_PAGE

    cursor.execute(base_query,
                   tuple(f"%{value}%" for value in search_query.values() if value) + (offset, RESULTS_PER_PAGE))
    runners = cursor.fetchall()


    visible_pages = 5  # Number of pages to display in pagination
    start_page = max(1, page - (visible_pages // 2))
    end_page = min(total_pages, start_page + visible_pages - 1)

    conn.close()
    return render_template('bib_management.html', runners=runners, page=page, total_pages=total_pages,
                           search_query=search_query, start_page=start_page, end_page=end_page)

@app.route('/add_bib_page', methods=['GET', 'POST'])
@login_required
def add_bib_page():
    conn = get_database_connection()
    cursor = conn.cursor()
    if request.method == 'POST':
        bib = request.form['BIB']
        name = request.form['Name']
        dob = request.form['DOB']
        age = request.form['Age']
        num = request.form['PhoneNumber']
        gender = request.form['Gender']
        national = request.form['National']
        dateRegister = request.form['DateRegister']
        distanceType = request.form['DistanceType']
        time = request.form['Time']
        pace = request.form['Pace']
        complete = request.form['Complete']
        finisherReceive = request.form['FinisherReceive']

        conn = get_database_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                    INSERT INTO dbo.Runners (BIB, Name, DOB, Age, PhoneNumber, Gender, [National], DateRegister, DistanceType, Time, Pace, Complete, FinisherReceive)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (bib, name, dob, age, num, gender, national, dateRegister, distanceType, time, pace, complete, finisherReceive))
            conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"Error adding BIB: {e}")
        finally:
            conn.close()
        return redirect(url_for('bib_management'))
    return render_template('add_bib.html')

@app.route('/update_bib/<string:bib>', methods=['GET', 'POST'])
@login_required
def update_bib(bib):
    conn = get_database_connection()
    cursor = conn.cursor()

    if request.method == 'POST':
        data = {key: request.form[key] for key in
                ['Name', 'DOB', 'Age', 'PhoneNumber', 'Gender', 'National', 'DateRegister', 'DistanceType', 'Time',
                 'Pace', 'Complete', 'FinisherReceive']}
        update_query = f"""
        UPDATE dbo.Runners 
        SET Name=%s, DOB=%s, Age=%s, PhoneNumber=%s, Gender=%s, [National]=%s, DateRegister=%s, DistanceType=%s, Time=%s, Pace=%s, Complete=%s, FinisherReceive=%s
        WHERE BIB=%s
        """
        cursor.execute(update_query, tuple(data.values()) + (bib,))
        conn.commit()
        conn.close()
        return redirect(url_for('bib_management'))

    cursor.execute("SELECT * FROM dbo.Runners WHERE BIB=%s", (bib,))
    runner = cursor.fetchone()
    conn.close()

    # Map the runner data to a dictionary for easier access in the template
    runner_data = {
        'BIB': runner[0],
        'Name': runner[1],
        'DOB': runner[2],
        'Age': runner[3],
        'PhoneNumber': runner[4],
        'Gender': runner[5],
        'National': runner[6],
        'DateRegister': runner[7],
        'DistanceType': runner[8],
        'Time': runner[9],
        'Pace': runner[10],
        'Complete': runner[11],
        'FinisherReceive': runner[12]
    }

    return render_template('update_bib.html', runner=runner_data)


@app.route('/delete_bib/<bib>', methods=['POST'])
@login_required
def delete_bib(bib):
    conn = get_database_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM dbo.Runners WHERE BIB = %s", (bib,))
    conn.commit()
    conn.close()

    return redirect(url_for('bib_management'))

@app.route('/image-gallery', methods=['GET','POST'])
@login_required
def image_gallery():
    search_bib = request.form.get('search_bib', '')

    # Get list of all images in the static/images directory
    all_images = [f for f in os.listdir(img_bucket) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    conn = get_database_connection()
    cursor = conn.cursor()

    bib_info = None
    if search_bib:
        cursor.execute(
            "SELECT Name, DOB, Age, Gender, PhoneNumber, [National], DateRegister, DistanceType, Time, Pace, Images, Notes FROM dbo.Runners WHERE BIB = %s",
            (search_bib,))
        result = cursor.fetchone()
        if result:
            bib_info = {
                'Name': result[0],
                'DOB': result[1],
                'Age': result[2],
                'Gender': result[3],
                'PhoneNumber': result[4],
                'National': result[5],
                'DateRegister': result[6],
                'DistanceType': result[7],
                'Time': result[8],
                'Pace': result[9],
                'Images': result[10].split() if result[10] else [],
                'Notes': result[11]
            }
            all_images = bib_info['Images']
        else:
            all_images = []

    # Pagination
    page = request.args.get('page', 1, type=int)
    per_page = 48
    total_pages = (len(all_images) + per_page - 1) // per_page
    start = (page - 1) * per_page
    end = start + per_page
    images = all_images[start:end]

    return render_template('image_gallery.html', images=images, page=page, total_pages=total_pages, search_bib=search_bib, bib_info=bib_info)

cleanup_uploads_folder(upload_folder)

atexit.register(cleanup_uploads_folder)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)