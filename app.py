#!/usr/bin/env python
import os
import shutil
import csv
from flask import Flask, render_template, request, \
    Response, send_file, redirect, url_for,flash
from camera import Camera
from flask import send_file, send_from_directory, safe_join, abort,session
from flask_sqlalchemy import SQLAlchemy


#mysql://root:''@localhost/attendance
app = Flask(__name__)
app.config["SECRET_KEY"]="abc"
app.config["SQLALCHEMY_DATABASE_URI"]="sqlite:///attendance.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
camera = None
db=SQLAlchemy(app)
# app.secret_key = "abc"


class Students(db.Model):
    roll_no=db.Column(db.Integer,primary_key=True)
    first_name=db.Column(db.String(200),nullable=False)
    last_name=db.Column(db.String(200),nullable=False)
    phone=db.Column(db.Integer(),nullable=False)


    def __init__(self,roll_no,first_name,last_name,phone):
        self.roll_no=roll_no
        self.first_name=first_name
        self.last_name=last_name
        self.phone=phone


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


@app.route('/index/', methods =["GET", "POST"])
def index():
    students=Students.query.all()
    return render_template('index_new.html',students=students)

@app.route('/teacher/', methods =["GET", "POST"])
def teacher_reg():
    return render_template('teacher.html')

@app.route('/subject/', methods =["GET", "POST"])
def subject_reg():
    return render_template('subject.html')


@app.route('/attendance', methods=['POST', 'GET'])
def attendance_records():
    return render_template('attendance.html')



@app.route('/insert',methods=["GET","POST"])
def insert():
    if request.method=="POST":
        # roll_no=request.form["roll"]
        # first_name=request.form["first_name"]
        # last_name=request.form["last_name"]
        # phone=request.form["phone"]

        roll_no=session.get("roll_no")
        first_name=session.get("first_name")
        last_name=session.get("last_name")
        phone=session.get("phone")

        students=Students(roll_no,first_name,last_name,phone)
        db.session.add(students)
        db.session.commit()
        flash("Student Added Sucessfully!!")

        return redirect(url_for("index"))



@app.route('/update',methods=["GET","POST"])
def update():
    if request.method=="POST":
        update_query=Students.query.get(request.form.get('id'))
        update_query.first_name=request.form['first_name']
        update_query.last_name=request.form['last_name']
        update_query.phone=request.form['phone']
        db.session.commit()
        flash("Student updated Sucessfully!!")
        return redirect(url_for('index'))


@app.route('/delete/<id>')
def delete(id):
    delete_student=Students.query.get(id)
    db.session.delete(delete_student)
    db.session.commit()
    flash("Student Deleted Sucessfully!!")
    return redirect(url_for('index'))





@app.route('/index_2/', methods =["GET", "POST"])
def indexing():
    if request.method=="POST":
        roll_no=request.form["roll"]
        first_name=request.form["first_name"]
        last_name=request.form["last_name"]
        phone=request.form["phone"]

        session["roll_no"]=roll_no
        session["first_name"]=first_name
        session["last_name"]=last_name
        session["phone"]=phone

        return render_template('index.html',roll_no=roll_no,first_name=first_name,last_name=last_name,phone=phone)

def gen(camera):
    while True:
        frame = camera.get_feed()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed/')
def video_feed():
    camera = get_camera()
    return Response(gen(camera),
        mimetype='multipart/x-mixed-replace; boundary=frame')

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
    
@app.route('/capture/image/<timestamp>', methods=['POST', 'GET'])
def show_capture(timestamp):
    path = stamp_file(timestamp)
    # print(path)
    roll_no=session.get("roll_no")
    first_name=session.get("first_name")
    last_name=session.get("last_name")
    phone=session.get("phone")
    return render_template('capture.html', path=path,roll_no=roll_no,first_name=first_name,last_name=last_name,phone=phone)




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
