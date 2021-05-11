#!/usr/bin/env python
import os
import shutil
import csv
from flask import Flask, render_template, request, \
    Response, send_file, redirect, url_for,flash
from camera import Camera
from flask import send_file, send_from_directory, safe_join, abort,session
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pickle
import cv2
import datetime
import xlwt
import xlrd
from facenet_pytorch import MTCNN
from xlwt import Workbook 
from matplotlib import pyplot
#from mtcnn.mtcnn import MTCNN
from matplotlib.patches import Rectangle
from keras.models import load_model
from matplotlib.patches import Circle
import cv2
import numpy as np
from sklearn.preprocessing import Normalizer
from scipy.spatial.distance import cosine
import os
from xlutils.copy import copy
# results=db.session.query(Students,Course,Classes).\
# ... select_from(Students).join(Course).join(Classes).all()


#mysql://root:''@localhost/attendance
app = Flask(__name__)
app.config["SECRET_KEY"]="abc"
app.config["SQLALCHEMY_DATABASE_URI"]="sqlite:///attendance.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
camera = None
db=SQLAlchemy(app)
# app.secret_key = "abc"


# lecs=db.Table('lecs',
#                 db.Column('class_id',db.Integer,db.ForeignKey('classes.class_id')),
#                 db.Column('lecture_id',db.Integer,db.ForeignKey('lectures.lecture_id')))


######################################### DATABASE ##################################################################
class Students(db.Model):
    roll_no=db.Column(db.String(200),primary_key=True)
    rank=db.Column(db.String(200),nullable=False)
    first_name=db.Column(db.String(200),nullable=False)
    last_name=db.Column(db.String(200),nullable=False)
    phone=db.Column(db.Integer(),nullable=False)
    attendance_id=db.relationship('Attendance_sys',backref='stud_attend')
    # room_id=db.Column(db.Integer,db.ForeignKey('classes.class_id'))
    course_sel=db.Column(db.Integer,db.ForeignKey('course.course_id'))

    


    # def __init__(self,roll_no,first_name,last_name,phone,room_id):
    #     self.roll_no=roll_no
    #     self.first_name=first_name
    #     self.last_name=last_name
    #     self.phone=phone
    #     self.room_id=room_id


class Timetable(db.Model):
    time_id=db.Column(db.Integer,primary_key=True)
    start_time=db.Column(db.String(200),nullable=False)
    end_time=db.Column(db.String(200),nullable=False)

class Course(db.Model):
    course_id=db.Column(db.Integer,primary_key=True)
    course_name=db.Column(db.String(200),unique=True,nullable=False)
    # stream=db.Column(db.String(200),nullable=False)
    courses=db.relationship('Students',backref='courses',cascade = "all,delete, delete-orphan")
    course_class=db.relationship('Classes',backref='course_class',cascade = "all,delete, delete-orphan")

class Classes(db.Model):
    class_id=db.Column(db.Integer,primary_key=True)
    classname=db.Column(db.String(200),unique=True,nullable=False)
    camera_name=db.Column(db.String(200),nullable=False)
    course_sel=db.Column(db.Integer,db.ForeignKey('course.course_id'))
    # roll=db.relationship('Students',backref='classroom')
    attendance_id=db.relationship('Attendance_sys',backref='cla_attend')
    # lectures=db.relationship('Lectures',secondary=lecs,backref=db.backref('subjects',lazy='dynamic'))

    # def __init__(self,class_id,classname):
    #     self.class_id=class_id
    #     self.classname=classname
    #     self.roll_no=roll_no
    #     self.lectures=lectures


class Lectures(db.Model):
    lecture_id=db.Column(db.Integer,primary_key=True)
    #Comment lecture_name and check how to show timings 
    lecture_name=db.Column(db.String(200),nullable=False)
    lecture_day=db.Column(db.String(200),nullable=False)
    lecture_time=db.Column(db.String(200),nullable=False)
    # lecture_datetime=db.Column(db.DateTime,default=datetime.utcnow())
    # lecture_start_time=db.Column(db.Time)
    # lecture_end_time=db.Column(db.Time)
    attendance_id=db.relationship('Attendance_sys',backref='lec_attend')

    # def __init__(self,lecture_id,class_id,lecture_date,lecture_start_time,lecture_end_time):
    #     self.lecture_id=lecture_id
    #     self.class_id=class_id
    #     self.lecture_date=lecture_date
    #     self.lecture_start_time=lecture_start_time
    #     self.lecture_end_time=lecture_end_time



class Attendance_sys(db.Model):
    attendance_id=db.Column(db.Integer,primary_key=True)
    roll_no=db.Column(db.Integer,db.ForeignKey('students.roll_no'))
    lecture_id=db.Column(db.Integer,db.ForeignKey('lectures.lecture_id'))
    class_id=db.Column(db.Integer,db.ForeignKey('classes.class_id'))
    present_absent=db.Column(db.Boolean, default=False, nullable=False)


    # def __init__(self,attendance_id,lecture_id,class_id,present_absent):
    #     self.attendance_id=attendance_id
    #     self.roll_no=roll_no
    #     self.lecture_id=lecture_id
    #     self.class_id=class_id
    #     self.present_absent=present_absent


########################################################################################################
###### Convert string to an Object
def str_to_class(str):
    return getattr(sys.modules[__name__], str)

    
def get_camera():
    global camera
    if not camera:
        camera = Camera()

    return camera


"""
@app.route('/')
def root():
    return redirect(url_for('index'))
"""
@app.route('/', methods =["GET", "POST"])
def image():
   if request.method == "POST":
       first_name = request.form.get("fname")
    
       last_name = request.form["lname"]
       print(last_name)
    #    session["a"]=first_name
    #    session["c"]=last_name

       
       #os.mkdir(str(first_name)+"_"+str(last_name))
       #os.chdir(str(first_name)+"_"+str(last_name))
       #return "Your name is "+first_name + last_name
       """with open('nameList.csv','w') as inFile:
            
            writer = csv.DictWriter(inFile, fieldnames=fieldnames)

            
            writer.writerow({'name': name, 'comment': comment})"""

       return render_template('index.html',first_name=first_name,last_name=last_name)
   return render_template('register.html')

######## Student Registration page
@app.route('/index/', methods =["GET", "POST"])
def index():
    # students=Students.query.all()
    # classrooms=Classes.query.all()
    # for c in classrooms:
    #     obj=Students.query.filter_by(room_id=c.class_id).first()
    # students=db.session.query(Students,Classes).join(Classes).all()
    course=Course.query.all()
    students=db.session.query(Students,Course).join(Course).all()
    return render_template('index_new.html',students=students,course=course)

######### Courses Registration Page
@app.route('/courses/', methods =["GET", "POST"])
def course_reg():
    courses=Course.query.all()
    return render_template('courses.html',courses=courses)

@app.route('/lectures/', methods =["GET", "POST"])
def lecture_reg():
    lectures=Lectures.query.all()
    return render_template('lectures.html',lectures=lectures)

######### Class Registration page
@app.route('/classes/', methods =["GET", "POST"])
def class_reg():
    classes=db.session.query(Classes,Course).join(Course).all()
    courses=Course.query.all()
    return render_template('classes.html',classes=classes,courses=courses)

########## Attendance Page
@app.route('/attendance', methods=['POST', 'GET'])
def attendance_records():
    return render_template('attendance.html')


######## Insert Student in the database
@app.route('/insert',methods=["GET","POST"])
def insert():
    if request.method=="POST":
        # roll_no=request.form["roll"]
        # first_name=request.form["first_name"]
        # last_name=request.form["last_name"]
        # phone=request.form["phone"]
        camera=get_camera()
        camera.stop_cam()
        roll_no=session.get("roll_no")
        rank=session.get("rank")
        first_name=session.get("first_name")
        last_name=session.get("last_name")
        phone=session.get("phone")
        course=session.get("course")
        # class_name=session.get("class_name")
        # class_name_1=Classes.query.filter_by(classname=class_name).first()

        # courses=Course(course_name=course)
        # db.session.add(courses)
        # db.session.commit()

        course_name=Course.query.filter_by(course_name=course).first()
        
        students=Students(roll_no=roll_no,rank=rank,first_name=first_name,last_name=last_name,phone=phone,courses=course_name)
        db.session.add(students)
        db.session.commit()
        flash("Student Added Sucessfully!!")
        return redirect(url_for("index"))


###### Insert course in the database
@app.route('/course_insert',methods=["POST","GET"])
def insert_course():
    if request.method=="POST":
        course_name=request.form["course_name"]

        stud_course=Course.query.all()

        for x in stud_course:
            if course_name==x.course_name:
                flash(1)
                return redirect(url_for("course_reg"))

        courses=Course(course_name=course_name)
        db.session.add(courses)
        db.session.commit()
        flash("Course Added Successfully!!")
        return redirect(url_for('course_reg'))

###### Insert class in the database
@app.route('/class_insert',methods=["POST","GET"])
def insert_class():
    if request.method=="POST":
        class_name=request.form["class_name"]
        camera_name=request.form["camera_name"]
        course_name=request.form["course"]

        stud_class=Classes.query.all()

        for x in stud_class:
            if class_name==x.classname:
                flash(1)
                return redirect(url_for("class_reg"))


        course_name=Course.query.filter_by(course_name=course_name).first()
        classes=Classes(classname=class_name,camera_name=camera_name,course_class=course_name)
        db.session.add(classes)
        db.session.commit()
        flash("Class Added Successfully!!")
        return redirect(url_for('class_reg'))


###### Update course in the database
@app.route('/update_course',methods=["GET","POST"])
def update_courses():
    if request.method=="POST":
        update_query_course=Course.query.get(request.form.get('id'))
        update_query_course.course_name=request.form["course"]
        db.session.commit()
        flash("Course updated Sucessfully!!!")
        return redirect(url_for('course_reg'))


###### Update class in the database
@app.route('/update_classes',methods=["GET","POST"])
def update_class():
    if request.method=="POST":
        update_query_class=Classes.query.get(request.form.get('id'))
        update_query_class.classname=request.form["class_name"]
        update_query_class.camera_name=request.form["camera_ip"]
        new_temp=request.form["course_name"]
        temp1=Course.query.filter_by(course_name=new_temp).first()
        update_query_class.course_sel=temp1.course_id

        db.session.commit()
        flash("Class updated Sucessfully!!!")
        return redirect(url_for('class_reg'))


###### Update student  in the database
@app.route('/update',methods=["GET","POST"])
def update():
    if request.method=="POST":
        update_query=Students.query.get(request.form.get('id'))
        #update_query_class=Students.query.get(request.form.get('classid'))
        update_query.rank=request.form['rank']
        update_query.first_name=request.form['first_name']
        update_query.last_name=request.form['last_name']
        update_query.phone=request.form['phone']
        temp=request.form["course"]
        # temp=request.form["class_name"]
        print(temp)
        temp1=Course.query.filter_by(course_name=temp).first()
        update_query.course_sel=temp1.course_id

        db.session.commit()
        flash("Student updated Sucessfully!!")
        return redirect(url_for('index'))

# @app.route('/update_lecture',methods=["GET","POST"])
# def update_lec():
#     if request.method=="POST":
#         # update_query=Lectures.query.get(request.form.get('id'))
#         # update_query.lecture_name=request.form['name']
#         # update_query.lecture_day=request.form['day']
#         # update_query.lecture_time=request.form['time']
#         # temp=request.form["class_name"]
#         # lecture_id=request.form.get('id')
#         lecture_id=request.form.get('id')
#         # Lectures.query.filter_by(lecture_id=lecture_id).delete()
#         delete_student=Lectures.query.get(lecture_id)
#         db.session.delete(delete_student)
#         db.session.commit()
#         lecture_name=request.form["name"]
#         lecture_day=request.form["day"]
#         lecture_time=request.form["time"]
#         class_name=request.form["class_name"]
#         lec=Lectures(lecture_id=lecture_id,lecture_name=lecture_name,lecture_day=lecture_day,lecture_time=lecture_time)
#         db.session.add(lec)
#         db.session.commit()
#         class_name_obj=Classes.query.filter_by(classname=class_name).first()
#         lec.subjects.append(class_name_obj)
#         db.session.commit()

#         # temp1=Classes.query.filter_by(classname=temp).first()
        
#         # for i,j in enumerate(update_query.subjects):
            

        
#         flash("Lecture updated Sucessfully!!")
#         return redirect(url_for('lecture_reg'))


##### Delete student in the database
@app.route('/delete/<id>')
def delete(id):
    delete_student=Students.query.get(id)
    course=delete_student.course_sel

    course_1=Course.query.filter_by(course_id=course).first()
    a=course_1.course_name
    b=delete_student.roll_no
    # shutil.rmtree("static/photo/"+str(a)+"/"+str(b)+"jpg",ignore_errors = True)
    os.remove("static/photo/"+str(a)+"/"+str(b)+".jpg")
    if os.path.isfile('static/embeddings/'+str(a)+'.dat'):
        with open('static/embeddings/'+str(a)+'.dat',"rb") as f:
            encoded = pickle.load(f)
        with open('static/embeddings/'+str(a)+'.dat', 'wb') as f1:
            del encoded[str(a)+"_"+str(b)]
            
            pickle.dump(encoded,f1)

    db.session.delete(delete_student)
    db.session.commit()
    flash("Student Deleted Sucessfully!!")
    return redirect(url_for('index'))

##### Delete class in the database
@app.route('/delete_classes/<id>')
def delete_class(id):
    delete_class=Classes.query.get(id)
    db.session.delete(delete_class)
    db.session.commit()
    flash("Class Deleted Sucessfully!!")
    return redirect(url_for('class_reg'))

##### Delete course in the database
@app.route('/delete_course/<id>')
def delete_courses(id):
    delete_course=Course.query.get(id)
    os.remove('static/embeddings'+'/'+str(delete_course.course_name)+'.dat')
    location="static/photo/"
    path=os.path.join(location,str(delete_course.course_name))
    print(path)
    shutil.rmtree(path,ignore_errors = True)
    db.session.delete(delete_course)
    db.session.commit()
    flash("Course Deleted Sucessfully!!")
    return redirect(url_for('course_reg'))


# @app.route('/delete_lecture/<id>')
# def delete_lec(id):
#     delete_lecture=Lectures.query.get(id)
#     db.session.delete(delete_lecture)
#     db.session.commit()
#     flash("Lecture Deleted Sucessfully!!")
#     return redirect(url_for('lecture_reg'))


# @app.route('/lecture_registered/',methods=["GET","POST"])
# def lec_complete_reg():
#     if request.method=="POST":
#         lecture_name=request.form["name"]
#         lecture_day=request.form["day"]
#         lecture_time=request.form["time"]
#         class_name=request.form["class_name"]
#         lec=Lectures(lecture_name=lecture_name,lecture_day=lecture_day,lecture_time=lecture_time)
#         db.session.add(lec)
#         db.session.commit()
#         class_name_obj=Classes.query.filter_by(classname=class_name).first()
#         lec.subjects.append(class_name_obj)
#         db.session.commit()
#         flash("Lecture Added Sucessfully!!")

#         return redirect(url_for('lecture_reg'))
        

#### After entering the details of the student session created to pass the contents of students to the next pages
@app.route('/index_2/', methods =["GET", "POST"])
def indexing():
    if request.method=="POST":
        roll_no=request.form["roll"]
        rank=request.form["rank"]
        first_name=request.form["first_name"]
        last_name=request.form["last_name"]
        phone=request.form["phone"]
        course=request.form["course"]
        # stream=request.form["stream"]
        # class_name=request.form["class_name"]

        studs=Students.query.all()

        for x in studs:
            if roll_no==x.roll_no:
                flash(1)
                return redirect(url_for("index"))
        

        session["roll_no"]=roll_no
        session["rank"]=rank
        session["first_name"]=first_name
        session["last_name"]=last_name
        session["phone"]=phone
        session["course"]=course
        # session["stream"]=stream
        
        # session["class_name"]=class_name
        # 
        return render_template('index.html',roll_no=roll_no,rank=rank,first_name=first_name,last_name=last_name,phone=phone,course=course)


### Camera access
def gen(camera):
    while True:
        frame = camera.get_feed()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            
@app.route('/video_feed/')
def video_feed():
    camera = get_camera()
    camera.start_cam()
    return Response(gen(camera),
        mimetype='multipart/x-mixed-replace; boundary=frame')


####Timestamp Creation
@app.route('/capture/')
def capture():
    #print(name)
    camera = get_camera()
    course=session.get("course")
    roll_no=session.get("roll_no")
    stamp,_ = camera.capture(course,roll_no)
    #print(filename)
    #f = ('%s.jpeg' % time.strftime("%Y%m%d-%H%M%S"))
    #camera.save('%s/%s' % ('None_None', f))

    return redirect(url_for('show_capture', timestamp=stamp))
    

"""    
@app.route('/uploads/<path:filename>', methods=['GET', 'POST'])


def download(filename):
    filename=str(request.args.get('first_name'))
    uploads = os.path.join(current_app.root_path, app.config['UPLOAD_FOLDER'])
    return send_from_directory(directory=uploads, filename=filename)
"""    
def stamp_file(timestamp):
    roll_no=session.get("roll_no")
    first_name=session.get("first_name")
    last_name=session.get("last_name")
    return 'photo/'+str(first_name)+'_'+str(last_name)+'/'+ timestamp +'.jpg'


##### Complete Details of students before inserting to the Database     
@app.route('/capture/image/<timestamp>', methods=['POST', 'GET'])
def show_capture(timestamp):
    path = stamp_file(timestamp)
    # print(path)
    roll_no=session.get("roll_no")
    rank=session.get("rank")
    first_name=session.get("first_name")
    last_name=session.get("last_name")
    phone=session.get("phone")
    # stream=session.get("stream")
    course=session.get("course")
    return render_template('capture.html', path=path,roll_no=roll_no,rank=rank,first_name=first_name,last_name=last_name,phone=phone,course=course)




"""
@app.route('/capture/image/<timestamp>', methods=['POST', 'GET'])
def show_capture(timestamp):
    
    path = stamp_file(timestamp)


    #email_msg = None
    #if request.method == 'POST':
        

    return render_template('capture.html',
        stamp=timestamp, path=path)

</form>
 <form method="GET" action="{{url_for('index')}}">
<button type="submit" > Take photo </button>
</form>"""

###################################################
####added by s

@app.route('/test')
def test():
    return render_template('program.html')



@app.route('/foo', methods=['POST'])
def foo():
    flag=False
    global video
    video =cv2.VideoCapture(0)
    
    # grab reddit data and write to csv
    program(flag)
    
    return jsonify({"message": "You have turned off the attendace system"})

@app.route('/new', methods=['POST'])
def new():
    flag=True
    # grab reddit data and write to csv
    program(flag)
    
    return redirect(url_for('test'))
@app.route('/highway', methods=['POST','GET'])
def highway():
    return redirect(url_for('index'))



recognition_t=0.6
confidence_t=0.99


encoder_model = 'facenet_keras.h5'

#detector=MTCNN()
detector=MTCNN()
face_encoder = load_model(encoder_model)
directory='static/embeddings'
encoded={}
for filename in os.listdir(directory):
    if filename.endswith(".dat"):
        if os.path.isfile('static/embeddings/'+str(filename)):
            with open('static/embeddings/'+str(filename),"rb") as f:
                e= pickle.load(f)
                encoded.update(e)

def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode

def get_face(img, box):
    [[x1, y1, width, height]] = box
    x1, y1 ,x2,y2= int(x1), int(y1),int(width),int(height)
    #x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)
 
def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

l2_normalizer = Normalizer('l2')
### collect daywise attendance by checking through a list of ppl

def mark_attendance_of_a_lec(a,t):
    workbook = xlwt.Workbook()  	 
    sheet = workbook.add_sheet(str(t.year)+"_"+str(t.month)+"_"+str(t.day)) 
    sheet.write(0,0,"Course")
    sheet.write(0,1,"Name")
    sheet.write(0,2,str(t.hour)+":"+str(t.minute))
    row = 1
    col = 0
    
    if len(a)>0:
        course=check_which_course(a)
        for person_name in encoded:
            print(person_name)
            #print(a)

            for x in range(0,len(a)):
                spl=str(a[x]).split('_')
                cou=spl[0]
                if person_name in a:
                    l=str(a[x]).split('_')
                    print(l)
                    if str(l[0])==str(course):
                        sheet.write(row, col,     str(l[0]))
    
                        sheet.write(row, col+1,     str(l[1]))
                        sheet.write(row,col+2,"P")
                if person_name not in a: 
                    l=str(person_name).split('_')
                    if course==cou:
                        sheet.write(row, col,     str(l[0]))
    
                        sheet.write(row, col+1,     str(l[1]))
                        sheet.write(row,col+2,"A")
                        
                
                row+=1
        #workbook.save("static/attendance/"+str(t.day)+"_"+str(t.month)+"_"+str(t.year)+"_"+str(t.hour)+":"+str(t.minute)+".xls")
        workbook.save(os.path.join('static/attendance', str(t.day)+"_"+str(t.month)+"_"+str(t.year)+"_"+str(t.hour)+":"+str(t.minute)+".xls"))
        
        
        print("Marked attendance")
    else:
        sheet.write(1,0,"No one is present")
        workbook.save("sample_class_1.xls") 
  
def check_which_course(a):
    number_of_s={}
    for x in range(0,len(a)):
        l=str(a[x]).split('_')
        if l[0] not in number_of_s:
            number_of_s[l[0]]=1
        else:
            number_of_s[l[0]]+=1
    course = max(number_of_s, key=number_of_s.get)
    return course
        



present_candidates=[]
fps_start_time = datetime.datetime.now()
classNames = []
with open('coco.names','r') as f:
    classNames = f.read().splitlines()
print(classNames)
thres = 0.5 # Threshold to detect object
nms_threshold = 0.2 #(0.1 to 1) 1 means no suppress , 0.1 means high suppress
weightsPath = "frozen_inference_graph.pb"
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

    

            
def program(flag):
    while True:
        check,frame=video.read()
        total_people=0
        t=datetime.datetime.now()
        #frame=sr.upsample(frame)
        #total_frames = total_frames + 1
        print(t.second)

        faces,_=detector.detect(frame)
        classIds, confs, bbox = net.detect(frame,confThreshold=thres)
        bbox = list(bbox)
        confs = list(np.array(confs).reshape(1,-1)[0])
        confs = list(map(float,confs))
    
        indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)
        if len(classIds) != 0:
        
            for i in indices:
                i = i[0]
            
                if classIds[i][0]==0:
                    total_people+=1
            
    
        #print(faces)
        if faces is not None:
            for person in faces:
                bounding_box=person
                face, pt_1, pt_2 = get_face(frame, [bounding_box])
                encode = get_encode(face_encoder, face,(160,160))
                encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
                name = 'unknown'
                distance = float("inf")
                for (db_name, db_enc) in encoded.items():
                    dist = cosine(db_enc, encode)
                    if dist < recognition_t and dist < distance:
                        name = db_name
                        distance = dist
                        if name not in present_candidates:
                            present_candidates.append(name)
        print(present_candidates)
        if t.second==59:
            mark_attendance_of_a_lec(present_candidates,t)     
        if flag:
           break  
     
    video.release()



if __name__ == '__main__':
    

    app.run(host='0.0.0.0', port=5000, debug=True)
