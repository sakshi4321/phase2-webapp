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
    course_name=db.Column(db.String(200),nullable=False)
    # stream=db.Column(db.String(200),nullable=False)
    courses=db.relationship('Students',backref='courses',cascade = "all,delete, delete-orphan")
    course_class=db.relationship('Classes',backref='course_class',cascade = "all,delete, delete-orphan")

class Classes(db.Model):
    class_id=db.Column(db.Integer,primary_key=True)
    classname=db.Column(db.String(200),nullable=False)
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
    first_name=session.get("first_name")
    last_name=session.get("last_name")
    stamp,_ = camera.capture(first_name,last_name)
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
if __name__ == '__main__':
    

    app.run(host='0.0.0.0', port=5000, debug=True)
