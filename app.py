from flask import Flask, render_template, request, flash, redirect, url_for, session
import ibm_db
import cv2
import numpy as np
import urllib.request
import os
from werkzeug.utils import secure_filename


app= Flask(__name__)
app.secret_key= b'dvno2698--34/'
conn = ibm_db.connect("database = bludb; hostname = b1bc1829-6f45-4cd4-bef4-10cf081900bf.c1ogj3sd0tgtu0lqde00.databases.appdomain.cloud; port = 32304; uid = gcn64999 ; password = B7wPQpEgWN2yRQQl; security=SSL; SSLServercertificate = DigiCertGlobalRootCA.crt ", " ", " ")

UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 
@app.route('/remove', methods=['POST'])
def remove_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        
        
        image = cv2.imread(file_path) 
        if image is None:
            print("Failed to load image")

        mask = np.zeros(image.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        rect = (50, 50, image.shape[1] - 50, image.shape[0] - 50)
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

        mask_2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        result = image * mask_2[:, :, np.newaxis]
        
        result_path = 'static/img/output_RemoveBackground.png'
        
        cv2.imwrite(result_path, result)
        
        return render_template('upload.html', filename=result_path)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
    
@app.route('/cools',methods=['POST'])
def cools():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        image = cv2.imread(file_path)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        glasses = cv2.imread('static/img/Cool.png', -1) 
        for (x, y, w, h) in faces:
            resized_glasses = cv2.resize(glasses, (w, int(h/2)))
            x_offset = x
            y_offset = y + int(h/4)
        for i in range(resized_glasses.shape[0]):
            for j in range(resized_glasses.shape[1]):
                if resized_glasses[i, j][2] != 0: 
                    image[y_offset + i, x_offset + j] = resized_glasses[i, j][:3]
                    result_path = 'static/img/output_Facefilter.png'
        cv2.imwrite(result_path, image)
        return render_template('cool.html', filename=result_path)
    else:
                    flash('Allowed image types are - png, jpg, jpeg, gif')
                    return redirect(request.url)
            
@app.route('/colors',methods=['POST'])
def colors():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        image = cv2.imread(file_path)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        glasses = cv2.imread('static/img/colorglasses.png', -1) 
        for (x, y, w, h) in faces:
            resized_glasses = cv2.resize(glasses, (w, int(h/2)))
            x_offset = x
            y_offset = y + int(h/4)
        for i in range(resized_glasses.shape[0]):
            for j in range(resized_glasses.shape[1]):
                if resized_glasses[i, j][2] != 0: 
                    image[y_offset + i, x_offset + j] = resized_glasses[i, j][:3]
                    result_path = 'static/img/output_Facefilter.png'
        cv2.imwrite(result_path, image)
        return render_template('color.html', filename=result_path)
    else:
                    flash('Allowed image types are - png, jpg, jpeg, gif')
                    return redirect(request.url)
                
@app.route('/swagg',methods=['POST'])
def swagg():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        image = cv2.imread(file_path)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        glasses = cv2.imread('static/img/swag.png', -1) 
        for (x, y, w, h) in faces:
            resized_glasses = cv2.resize(glasses, (w, int(h/2)))
            x_offset = x
            y_offset = y + int(h/4)
        for i in range(resized_glasses.shape[0]):
            for j in range(resized_glasses.shape[1]):
                if resized_glasses[i, j][2] != 0: 
                    image[y_offset + i, x_offset + j] = resized_glasses[i, j][:3]
                    result_path = 'static/img/output_Facefilter.png'
        cv2.imwrite(result_path, image)
        return render_template('swag.html', filename=result_path)
    else:
                    flash('Allowed image types are - png, jpg, jpeg, gif')
                    return redirect(request.url)
                
@app.route('/girlys',methods=['POST'])
def girlys():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        image = cv2.imread(file_path)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        glasses = cv2.imread('static/img/Girly.png', -1) 
        for (x, y, w, h) in faces:
            resized_glasses = cv2.resize(glasses, (w, int(h/2)))
            x_offset = x
            y_offset = y + int(h/4)
        for i in range(resized_glasses.shape[0]):
            for j in range(resized_glasses.shape[1]):
                if resized_glasses[i, j][2] != 0: 
                    image[y_offset + i, x_offset + j] = resized_glasses[i, j][:3]
                    result_path = 'static/img/output_Facefilter.png'
        cv2.imwrite(result_path, image)
        return render_template('girly.html', filename=result_path)
    else:
                    flash('Allowed image types are - png, jpg, jpeg, gif')
                    return redirect(request.url)
                
@app.route('/pixels',methods=['POST'])
def pixels():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        low_res = cv2.imread(file_path)
        scale_factor = 8
        high_res = cv2.resize(low_res, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        result_path = 'static/img/output_Pixelperfect.png'
        cv2.imwrite(result_path, high_res)
        return render_template('upload.html', filename=result_path)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
@app.route('/enhance',methods=['POST'])
def enhance():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        image = cv2.imread(file_path) 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        result_path = 'static/img/output_enhanced.jpg'
        cv2.imwrite(result_path, enhanced)
        return render_template('upload.html', filename=result_path)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
    
@app.route('/cartoons',methods=['POST'])
def cartoon():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        image = cv2.imread(file_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        result_path = 'static/img/output_cartoon.jpg'
        cv2.imwrite(result_path, sharpened)
        return render_template('cartoon.html', filename=result_path)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
           
@app.route("/signin",methods = ['GET','POST'])
def signin():
    global u_email
    if request.method == 'POST':
        u_email = request.form['nameField']
        u_pass = request.form['pwd']
        print("The Username of the email : {} and password : {}". format(u_email, u_pass))
        sql  = "SELECT * from REGISTER_DS WHERE EMAIL =  ? AND PWD = ?"
        stmt = ibm_db.prepare(conn, sql)
        ibm_db.bind_param(stmt, 1, u_email)
        ibm_db.bind_param(stmt, 2, u_pass)
        ibm_db.execute(stmt)
        info = ibm_db.fetch_assoc(stmt)
        print(info)
        if info : 
            session['id'] = True
            session['email'] = u_email

            return render_template("menu.html")
        else:
            msg_w = "Check the Username and Password you have entered"
            return render_template("signin.html", msg_w = msg_w ) 
            
    return render_template("signin.html")

@app.route("/SignUp", methods=['GET', 'POST'])
def u_login():
    global u_email
    if request.method == 'POST':
        u_name = request.form['user']
        u_email = request.form['nameField']
        u_pass = request.form['pwd']
        print("Entered details for registation are : " ,u_name, u_email, u_pass)
        sql  = "SELECT * from REGISTER_DS WHERE USER = ? AND  EMAIL =  ? AND PWD = ?"
        stmt = ibm_db.prepare(conn, sql)
        ibm_db.bind_param(stmt, 1, u_name)
        ibm_db.bind_param(stmt, 2, u_email)
        ibm_db.bind_param(stmt, 3, u_pass)
        ibm_db.execute(stmt)
        info = ibm_db.fetch_assoc(stmt)
        print("info we got from the table : " , info)
        if info : 
            msg = "Your have been already registered : Kindly LOGIN"
            return render_template("login.html", msg = msg )
        else: 
            sql = "INSERT into REGISTER_DS VALUES (?, ?, ?)"
            stmt = ibm_db.prepare(conn, sql)
            ibm_db.bind_param(stmt, 1 , u_name)
            ibm_db.bind_param(stmt, 2 , u_email)
            ibm_db.bind_param(stmt, 3 , u_pass)
            ibm_db.execute(stmt)
            msg_r = "your are successfully registered : kindly LOGIN"
            return render_template("login.html", msg_r = msg_r)
    return render_template("login.html")
        
@app.route("/")
def login():
    return render_template("home.html")
@app.route("/about")
def about():
    return render_template("about.html")
@app.route("/service")
def service():
    return render_template("service.html")
@app.route("/contact")
def contact():
    return render_template("contact.html")
@app.route("/home")
def home():
    return render_template("home.html")
@app.route("/logout")
def logout():
    return render_template("home.html")
@app.route("/upload")
def upload():
    return render_template("upload.html")
@app.route("/color")
def color_file():
    return render_template("color.html")
@app.route("/pixel")
def pixel_file():
    return render_template("pixel.html")
@app.route("/cartoon")
def cartoon_file():
    return render_template("cartoon.html")
@app.route("/enhancement")
def enhancement_file():
    return render_template("enhancement.html")
@app.route("/menu")
def back():
    return render_template("menu.html")
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")
@app.route("/glasses")
def glasses():
    return render_template("glasses.html")
@app.route("/color")
def color():
    return render_template("color.html")
@app.route("/cool")
def cool():
    return render_template("cool.html")
@app.route("/swag")
def swag():
    return render_template("swag.html")
@app.route("/girly")
def girly():
    return render_template("girly.html")



if __name__=="__main__":
    app.run(debug=True)
    
    
    