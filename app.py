#!/usr/bin/env python
import os
import shutil
import csv
from flask import Flask, render_template, request, \
    Response, send_file, redirect, url_for,flash
from camera import Camera
from flask import send_file, send_from_directory, safe_join, abort,session,make_response 
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pickle
import cv2
import datetime
import pandas as pd
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


class Teachers(db.Model):
    roll_no=db.Column(db.String(200),primary_key=True)
    rank=db.Column(db.String(200),nullable=False)
    first_name=db.Column(db.String(200),nullable=False)
    last_name=db.Column(db.String(200),nullable=False)
    phone=db.Column(db.Integer(),nullable=False)
    attendance_id=db.relationship('Attendance',backref='attendance_id',cascade="all,delete,delete-orphan")
    # track_id=db.relationship('Track',backref='track_id',cascade="all,delete,delete-orphan")
    # room_id=db.Column(db.Integer,db.ForeignKey('classes.class_id'))

class Attendance(db.Model):
    sr=db.Column(db.Integer,primary_key=True,autoincrement=True)
    in_time=db.Column(db.String(200),nullable=False)
    out_time=db.Column(db.String(200),nullable=False)
    date_att=db.Column(db.String(200),nullable=False)
    teacher_sel=db.Column(db.String(200),db.ForeignKey('teachers.roll_no'))



class Track(db.Model):
    sr=db.Column(db.Integer,primary_key=True,autoincrement=True)
    teach_track=db.Column(db.String(200),nullable=False)
    in_out=db.Column(db.Boolean,default=False)





######################################### DATABASE ##################################################################
def get_camera():
    global camera
    if not camera:
        camera = Camera()

    return camera


@app.route('/',methods=["GET","POST"])
def teacher_reg():
    # if request.method=="GET":
    teachers=Teachers.query.all()
    return render_template("index_new.html",teachers=teachers)



@app.route('/index_2/', methods =["GET", "POST"])
def indexing():
    if request.method=="POST":
        roll_no=request.form["roll"]
        rank=request.form["rank"]
        first_name=request.form["first_name"]
        last_name=request.form["last_name"]
        phone=request.form["phone"]
       
        # stream=request.form["stream"]
        # class_name=request.form["class_name"]

        teach=Teachers.query.all()

        for x in teach:
            if roll_no==x.roll_no:
                flash(1)
                return redirect(url_for("teacher_reg"))
        

        session["roll_no"]=roll_no
        session["rank"]=rank
        session["first_name"]=first_name
        session["last_name"]=last_name
        session["phone"]=phone
        

        # 
        return render_template('index.html',roll_no=roll_no,rank=rank,first_name=first_name,last_name=last_name,phone=phone)


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

@app.route('/capture/')
def capture():
    #print(name)
    camera = get_camera()
    # course=session.get("course")
    roll_no=session.get("roll_no")
    stamp,_ = camera.capture(roll_no)
    #print(filename)
    #f = ('%s.jpeg' % time.strftime("%Y%m%d-%H%M%S"))
    #camera.save('%s/%s' % ('None_None', f))

    return redirect(url_for('show_capture', timestamp=stamp))


def stamp_file(timestamp):
    roll_no=session.get("roll_no")
    return 'photo/'+str(roll_no)+'.jpg'


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
    return render_template('capture.html', path=path,roll_no=roll_no,rank=rank,first_name=first_name,last_name=last_name,phone=phone)


@app.route('/insert',methods=["GET","POST"])
def insert():
    if request.method=="POST":

        camera=get_camera()
        camera.stop_cam()
        roll_no=session.get("roll_no")
        rank=session.get("rank")
        first_name=session.get("first_name")
        last_name=session.get("last_name")
        phone=session.get("phone")
        
        teachers=Teachers(roll_no=roll_no,rank=rank,first_name=first_name,last_name=last_name,phone=phone)

        db.session.add(teachers)
        # db.session.commit()

        # teach=Teachers.query.filter_by(roll_no=roll_no).first()

        track=Track(teach_track=roll_no,in_out=False)
        db.session.add(track)
        db.session.commit()
        
        flash("Teacher Added Sucessfully!!")
        return redirect(url_for("teacher_reg"))

@app.route('/update',methods=["GET","POST"])
def update():
    if request.method=="POST":
        update_query=Teachers.query.get(request.form.get('id'))
        #update_query_class=Students.query.get(request.form.get('classid'))
        update_query.rank=request.form['rank']
        update_query.first_name=request.form['first_name']
        update_query.last_name=request.form['last_name']
        update_query.phone=request.form['phone']
  
        # temp=request.form["class_name"]

        db.session.commit()
        flash("Teacher updated Sucessfully!!")
        return redirect(url_for('teacher_reg'))



@app.route('/delete/<id>')
def delete(id):
    delete_student=Teachers.query.get(id)
    a=delete_student.roll_no
    # shutil.rmtree("static/photo/"+str(a)+"/"+str(b)+"jpg",ignore_errors = True)
    os.remove("static/photo/"+str(a)+".jpg")
    if os.path.isfile('static/embeddings/'+str(a)+'.dat'):
        with open('static/embeddings/'+str(a)+'.dat',"rb") as f:
            encoded = pickle.load(f)
        with open('static/embeddings/'+str(a)+'.dat', 'wb') as f1:
            del encoded[str(a)]
            
            pickle.dump(encoded,f1)

    db.session.delete(delete_student)
    tr=Track.query.filter_by(teach_track=delete_student.roll_no).first()
    db.session.delete(tr)
    db.session.commit()
    flash("Teacher Deleted Sucessfully!!")
    return redirect(url_for('teacher_reg'))


@app.route('/attendance',methods=['GET','POST'])
def attendance_sys():
    att=db.session.query(Attendance,Teachers).join(Teachers).all()
    return render_template('attendance.html',att=att)

@app.route('/insert_attendee',methods=["GET","POST"])
def ins_attendee():
    roll_no=request.form["roll"]
    # rank=request.form["rank"]
    # first_name=request.form["first_name"]
    # last_name=request.form["last_name"]
    date_att=request.form["date"]
    in_time=request.form["in_time"]
    out_time=request.form["out_time"]

    # teach=Teachers.query.all()

    teach=Teachers.query.filter_by(roll_no=roll_no).first()

    if teach is None:
        flash(1)
    else:
        att=Attendance(in_time=in_time,out_time=out_time,date_att=date_att,attendance_id=teach)
        db.session.add(att)
        db.session.commit()

    return redirect(url_for('attendance_sys'))


@app.route('/delete_attendance/<id>')
def del_att(id):
    # delete_teach=Teachers.query.get(id)
    final_del=Attendance.query.filter_by(sr=id).first()
    db.session.delete(final_del)
    db.session.commit()
    flash("Attendance Deleted Sucessfully!!")
    return redirect(url_for('attendance_sys'))

@app.route('/update_attendance',methods=["GET","POST"])
def update_att():
    update_teach=Teachers.query.get(request.form["id"])
    final_update=Attendance.query.filter_by(teacher_sel=update_teach.roll_no).first()
    final_update.in_time=request.form["in_time"]
    final_update.out_time=request.form["out_time"]
    final_update.date_att=request.form["date"]
    db.session.commit()
    flash("Attendance Update Successfully !!")
    return redirect(url_for('attendance_sys'))
#######################start and stop button
@app.route('/test')
def test():
    return render_template('program.html')

def to_dict(row):
    if row is None:
        return None

    rtn_dict = dict()
    keys = row.__table__.columns.keys()
    for key in keys:
        #print(key)
        rtn_dict[key] = getattr(row, key)
        #print(rtn_dict)
    return rtn_dict
##################program to get date input from user 
@app.route('/excel', methods =["GET", "POST"])
def excel():
    
    if request.method == "POST":
       #get start date and end date
       start = request.form.get("start")
       
       end = request.form.get("end") 
       
       qry = Attendance.query.filter(Attendance.date_att.between(start, end)).all()
       data_list = [to_dict(item) for item in qry]
       df = pd.DataFrame(data_list)
       #print(df)
       #a=df.columns
       df.drop('sr',inplace=True,axis=1)
       #df.drop('primkey',inplace=True,axis=1)
       #df.drop('lecture_no',inplace=True,axis=1)
       #s = df.groupby(['id_a']).cumcount()

       #df1 = df.set_index(['id_a', s]).unstack().sort_index(level=1, axis=1)
       #df1.columns = [f'{x}{y}' for x, y in df1.columns]
       #df1 = df1.reset_index()
       df.columns = ['in time', 'out time','date','ID']
       first_column = df.pop('ID')
       df.insert(0, 'ID', first_column)
       print (df)

       #print(df.date.unique())
       #print(df)
       resp = make_response(df.to_csv(index=False))
       resp.headers["Content-Disposition"] = "attachment; filename=export.csv"
       resp.headers["Content-Type"] = "text/csv"
       return resp
       
       #for i in qry:
       #print(i.date)
    return redirect(url_for("attendance_sys"))

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


# print

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

# def mark_attendance_of_a_lec(a,t):
#     workbook = xlwt.Workbook()  	 
#     sheet = workbook.add_sheet(str(t.year)+"_"+str(t.month)+"_"+str(t.day)) 
#     sheet.write(0,0,"Course")
#     sheet.write(0,1,"Name")
#     sheet.write(0,2,str(t.hour)+":"+str(t.minute))
#     row = 1
#     col = 0
    
#     if len(a)>0:
#         course=check_which_course(a)
#         for person_name in encoded:
#             print(person_name)
#             #print(a)

#             for x in range(0,len(a)):
#                 spl=str(a[x]).split('_')
#                 cou=spl[0]
#                 if person_name in a:
#                     l=str(a[x]).split('_')
#                     print(l)
#                     if str(l[0])==str(course):
#                         sheet.write(row, col,     str(l[0]))
    
#                         sheet.write(row, col+1,     str(l[1]))
#                         sheet.write(row,col+2,"P")
#                 if person_name not in a: 
#                     l=str(person_name).split('_')
#                     if course==cou:
#                         sheet.write(row, col,     str(l[0]))
    
#                         sheet.write(row, col+1,     str(l[1]))
#                         sheet.write(row,col+2,"A")
                        
                
#                 row+=1
#         #workbook.save("static/attendance/"+str(t.day)+"_"+str(t.month)+"_"+str(t.year)+"_"+str(t.hour)+":"+str(t.minute)+".xls")
#         workbook.save(os.path.join('static/attendance', str(t.day)+"_"+str(t.month)+"_"+str(t.year)+"_"+str(t.hour)+":"+str(t.minute)+".xls"))
        
        
#         print("Marked attendance")
#     else:
#         sheet.write(1,0,"No one is present")
#         workbook.save("sample_class_1.xls") 
  
# def check_which_course(a):
#     number_of_s={}
#     for x in range(0,len(a)):
#         l=str(a[x]).split('_')
#         if l[0] not in number_of_s:
#             number_of_s[l[0]]=1
#         else:
#             number_of_s[l[0]]+=1
#     course = max(number_of_s, key=number_of_s.get)
#     return course
        



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




def record_attend(flag):
    
    if flag == True:
        present_candidates=[]
        while True:
            now=datetime.datetime.now()
            current_date=now.strftime("%Y-%m-%d")
            print(current_date)
            everyone=db.session.query(Attendance,Teachers).join(Teachers).all()
            check,frame=video.read()
            total_people=0
            t=datetime.datetime.now()
            flag=0
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

                            if t.second==58:
                                print(present_candidates)
                                present_candidates=[]

                            elif name not in present_candidates and t.second!=58:
                                present_candidates.append(name)
                                track=Track.query.filter_by(teach_track=name).first()
                                for i,j in everyone:
                                    
                                    if j.roll_no == name and i.date_att==current_date:
                                        if track.in_out==True:
                                            teach=Teachers.query.filter_by(roll_no=j.roll_no).first()
                                            teach_attend=Attendance.query.filter_by(teacher_sel=teach.roll_no).all()
                                            for q in teach_attend:
                                                if q.out_time=="--":
                                                    time = now.strftime("%H:%M")
                                                    q.out_time=time
                                                    track.in_out = False
                                            flag=1
                                            break


                                        elif track.in_out==False:
                                            teach=Teachers.query.filter_by(roll_no=name).first()
                                            time = now.strftime("%H:%M")
                                            att=Attendance(in_time=time,out_time="--",date_att=current_date,attendance_id=teach)
                                            db.session.add(att)
                                            track.in_out= True
                                            flag=1
                                            break


                                        else:
                                            teach=Teachers.query.filter_by(roll_no=j.roll_no).first()
                                            teach_attend=Attendance.query.filter_by(teacher_sel=teach.roll_no).first()
                                            time = now.strftime("%H:%M")
                                            teach_attend.out_time=time
                                            track.in_out= False
                                            flag=1
                                            break

                                if flag==0:
                                    teach=Teachers.query.filter_by(roll_no=name).first()
                                    time = now.strftime("%H:%M")
                                    att=Attendance(in_time=time,out_time="--",date_att=current_date,attendance_id=teach)
                                    track1=Track.query.filter_by(teach_track=name).first()
                                    track1.in_out= True
                                    db.session.add(att)

                                db.session.commit()




                                



            

            # print(present_candidates)
            # if t.second==59:
            #     mark_attendance_of_a_lec(present_candidates,t)     
            # if flag:
            #     break  
    else:
        video.release()

    video.release()


@app.route('/highway', methods=['POST','GET'])
def highway():
    return redirect(url_for('attendance_sys'))
@app.route('/start_attendance',methods=["POST"])
def start_attendance():
    flag=True
    global video
    video=cv2.VideoCapture(0)
    flash("Attendance system started")
    record_attend(flag)

    
    return redirect(url_for('attendance_sys'))


@app.route('/stop_attendance',methods=["POST"])
def stop_attendance():
    flag=False
    record_attend(flag)

    flash("Attendance system stopped")
    return redirect(url_for('attendance_sys'))




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5020, debug=True)
