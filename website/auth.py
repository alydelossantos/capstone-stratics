import string
import psycopg2
import pandas as pd
import numpy as np
import random
import os
from os.path import join
import secrets
import smtplib
import sqlalchemy 
from PIL import Image
from flask import Flask
import base64
import mimetypes
from .extensions import db
from flask import Blueprint, render_template, request, flash, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func, create_engine
from .models import User, Data, Sampledata, Strategies, Samplestrategies, Contact, Task
from flask_login import login_user, login_required, logout_user, current_user
from werkzeug.utils import secure_filename
from website import mail
from datetime import datetime, date
from itsdangerous import URLSafeTimedSerializer
from flask_mail import Mail, Message
from email.message import EmailMessage
from email import encoders
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.message import EmailMessage
from wtforms.validators import DataRequired
from wtforms import StringField
from .forms import RequestResetForm, ResetPasswordForm

auth = Blueprint('auth', __name__)
 
kfull = "Kalibo Cable Television Network, Inc."
knoc = "Kalibo Cable Television Network Inc."
knop = "Kalibo Cable Television Network, Inc"
knob = "Kalibo Cable Television Network Inc"
knoinc = "Kalibo Cable Television Network"
knonet = "Kalibo Cable Television"
knotel = "Kalibo Cable"
knocable = "Kalibo"
abbrenoinc = "KCTN"


conn = psycopg2.connect("postgresql://ogegtvpsnmnlhf:a96fbd418afd0730555278f2bafce6f7cbedeafe42bd3a8cc47e802e0a1c7916@ec2-52-21-136-176.compute-1.amazonaws.com:5432/db0j0m4lc1ku1u")
conn.autocommit =True
cur = conn.cursor()

# Landing Page
#About Page
@auth.route('/about')
def about():
    return render_template("about.html", user= current_user)
    
#pripo page
@auth.route('/privacy-policy') 
def pripo():
    return render_template("privacypolicy.html", user= current_user)

#t&c page 
@auth.route('/terms-conditions') 
def tc():
    return render_template("termsconditions.html", user= current_user)

@auth.route('/feature-dashboard') #feature dashboard
def fdash():
    return render_template("feature-dashboard.html", user= current_user)

@auth.route('/feature-custman') #feature custman
def fcustman():
    return render_template("feature-custman.html", user= current_user)

@auth.route('/feature-strategies') #feature strategies
def fstrategies():
    return render_template("feature-strategies.html", user= current_user)

@auth.route('/feature-email') #feature email marketing
def femail():
    return render_template("feature-email.html", user= current_user)

#CONTACT page
@auth.route('/contact', methods=["GET", "POST"]) 
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
 
        cont = Contact(name=name, email=email, message=message)
        db.session.add(cont)
        db.session.commit()
    
        flash("Inquiry Successfully Sent!")
        
    return render_template("contact.html", user= current_user)

#SIGNIN page   
@auth.route('/sign-in', methods=["GET", "POST"]) 
def signin():  
    if request.method == "POST" :
        email = request.form.get("email")
        password = request.form.get("password")
        
        user = User.query.filter_by(email=email).first()
        if user:
            if user.password == password:
                if user.user_type == "user":
                    if user.email_confirmed == True:
                        if user.cname.lower() == kfull.lower() or user.cname.lower() == knoc.lower() or user.cname.lower() == knob.lower() or user.cname.lower() == knop.lower() or user.cname.lower() == knoinc.lower() or user.cname.lower() == knonet.lower() or user.cname.lower() == knotel.lower() or user.cname.lower() == knocable.lower() or user.cname.lower() == abbrenoinc.lower():
                            user.cname = "Kalibo Cable"
                            user.ccode =  "11A392O"
                            user.user_status = True
                            db.session.add(user)
                            db.session.commit()
                            return redirect(url_for("auth.checkcode"))
                        else:
                            login_user(user, remember=True)
                            user.user_status = True
                            db.session.add(user)
                            db.session.commit()
                            return redirect(url_for("views.home"))
                    else:
                      flash("Please confirm your account!", category="error")
                else:
                    flash("You do not have an access to this webpage.", category="error")
            else:
                flash("Password Incorrect. Please try again", category="error")
        else:
            flash("Email does not exists.", category="error")
        
    return render_template("signin.html", user= current_user)

#SIGNNIN CHECK CODE page   
@auth.route('/sign-in/check-code', methods=["GET", "POST"])
def checkcode():
    if request.method == "POST" :
        email = request.form.get("email")
        ccode = request.form.get("ccode")
        user_status = request.form.get("user_status")
        
        user = User.query.filter_by(email=email).first()

        if user.ccode == ccode:
            login_user(user, remember=True)
            return redirect(url_for("views.home"))
        else:
            flash("Incorrect Company Code", category="error")
    return render_template("check_code.html", user= current_user)
          
@auth.route('/sign-out') #signout page
@login_required
def signout():
    if current_user:
        current_user.user_status = False
        db.session.commit()
        logout_user()
    return redirect(url_for("views.landing"))

#SIGNUP page
@auth.route('/sign-up', methods=["GET", "POST"]) #signup page
def signup():
    if request.method == "POST" :
        fname = request.form.get("fname")
        lname = request.form.get("lname")
        uname = request.form.get("uname")
        email = request.form.get("email")
        cname = request.form.get("cname")
        password = request.form.get("password")
        user_type = request.form.get("user_type")

        user = User.query.filter_by(email=email).first()
        usern = User.query.filter_by(uname=uname).first()
        if user:
            flash("Email already exists.", category="error")
        elif usern:
            flash("Username already exists.", category="error")
        elif len(password) < 8:
            flash("Password must contain 8 characters.", category="error")
        elif len(fname) < 2:
            flash("Please input a valid name.", category="error")
        elif len(lname) < 2:
            flash("Please input a valid name.", category="error")
        else:
            new_user = User(fname=fname, lname=lname, uname=uname, email=email, cname=cname, password=password, user_type=user_type)
            db.session.add(new_user)
            db.session.commit()
            send_confirmation_email(new_user.email)
            flash("Thank you for registering! Please check your email to confirm your account.", category="success")
            return redirect(url_for("auth.signin"))
    return render_template("signup.html", user= current_user)
 
def send_email(subject, recipients, html_body):
    msg = Message(subject, recipients=recipients)
    msg.html = html_body
    mail.send(msg)
      
def send_confirmation_email(user_email):
    confirm_serializer = URLSafeTimedSerializer('asdfghjkl')
    
    confirm_url = url_for(
        'auth.confirm_email', 
        token=confirm_serializer.dumps(user_email, salt='email-confirmation-salt'), 
        _external = True)
    
    html = render_template('confirmemail.html', confirm_url=confirm_url)
    
    send_email('STRATICS EMAIL CONFIRMATION', [user_email], html)
   
@auth.route('/sign-up/confirm/<token>')
def confirm_email(token):
    try:
        confirm_serializer = URLSafeTimedSerializer('asdfghjkl')
        email = confirm_serializer.loads(token, salt='email-confirmation-salt', max_age=3600)
    except:
        flash('The confirmation link is invalid.', category="error")
        return redirect(url_for('auth.signin'))
        
    user = User.query.filter_by(email=email).first()
    
    if user.email_confirmed:
        flash('Your account has been confirmed.')
    else:
        user.email_confirmed = True 
        user.email_confirmed_on = datetime.now()
        db.session.add(user)
        db.session.commit()
    login_user(user, remember=True)   
    return redirect(url_for('auth.signin'))
    return render_template(user= current_user)

  
# Home Page

# Side Bar
def sidebarpic():
    image_file = url_for('static', filename='images/' + current_user.image_file)
    return render_template("base.html", user= current_user, image_file = image_file)

# Customer Management
@auth.route('/customer-management', methods=["GET", "POST"]) 
@login_required
def custman():
    if current_user.explore == "customer" or current_user.explore == "empty":
        if current_user.cname == "Kalibo Cable":
            all_data = Data.query.all()
            
            image_file = url_for('static', filename='images/' + current_user.image_file)
            return render_template("custman.html", user= current_user, datas=all_data, image_file = image_file)		
    elif current_user.explore == "sample":
        all_data = Sampledata.query.all()
        
        image_file = url_for('static', filename='images/' + current_user.image_file)
        return render_template("custman.html", user= current_user, sampledatas=all_data, image_file = image_file)

@auth.route('/customer-management/insert', methods = ['POST'])
@login_required
def insert():
    if current_user.cname == "Kalibo Cable":
        if request.method == 'POST':
            account_no = request.form['account_no']
            subscriber = request.form['subscriber']
            address = request.form['address']
            zone = request.form['zone']
            services = request.form['services']
            monthly = request.form['monthly']
            collector = request.form['collector']
            amount_paid = request.form['amount_paid']
            ref_no = request.form['ref_no']
            date_paid = request.form['date_paid']
            category = request.form['category']
            activation_date = request.form['activation_date']
            disconnection_date = request.form['disconnection_date']
            reactivation_date = request.form['reactivation_date']
            
            if disconnection_date == "" and reactivation_date == "":
                disconnection_date = None
                reactivation_date = None
            elif disconnection_date != "" and reactivation_date == "":
                reactivation_date = None
                disconnection_date = request.form['disconnection_date']
            elif disconnection_date != "" and reactivation_date != "":
                reactivation_date = request.form['reactivation_date']
                disconnection_date = request.form['disconnection_date']
            
            if activation_date != None and disconnection_date == None:
                status = "Active"
            elif activation_date != None and disconnection_date != None:
                status = "Disconnected"
            elif activation_date != None and disconnection_date != None and reactivation_date != None:
                status = "Disconnected"

            if disconnection_date == None:
                churn = 0
            else:
                churn = 1
		        
            row = Data.query.count()
            count = Data.query.filter(Data.id >= row).count()
            if count >= 1:
                id = row + count
                datas = Data(id=id)
            
            total_paid = 0
            total_paid = float(amount_paid) + float(total_paid)
            last_modified_on = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(disconnection_date)
            datas = Data(account_no=account_no, subscriber=subscriber, address=address, zone=zone, services=services, monthly=monthly,
					collector=collector, status=status, amount_paid=amount_paid, total_paid=total_paid, ref_no=ref_no, date_paid=date_paid, category=category, activation_date=activation_date,
					disconnection_date=disconnection_date, reactivation_date=reactivation_date, last_modified_on=last_modified_on, churn=churn)
            db.session.add(datas)
            db.session.commit()
            
            flash("Customer Record Added Successfully")
            
            return redirect(url_for('auth.custman'))
            #return render_template(ran=ran)

@auth.route('/import', methods = ['GET','POST'])
@login_required
def importcsv():
    if current_user.cname == "Kalibo Cable":
        if request.method == 'POST':
            if request.files['csv']:
                csv_file = save_import(request.files['csv'])
                current_user.csv = csv_file  
            conn.commit()

            col = ['id','account_no', 'subscriber', 'address', 'zone', 'services', 'monthly', 'collector', 'status', 'amount_paid', 'total_paid', 'ref_no', 'date_paid', 'category', 'activation_date', 'disconnection_date', 'reactivation_date', 'last_modified_on', 'churn']
            url = "https://raw.githubusercontent.com/alydelossantos/capstone-stratics/main/website/static/file/kalibo2018.csv"          
            records = pd.read_csv(url,index_col=0, header=0)
            print(records)
            records['last_modified_on'].replace({np.nan:datetime.now()}, inplace=True)
            print(records['last_modified_on'])
            for i, row in records.iterrows():

                sql = "INSERT INTO data (account_no, subscriber, address, zone, services, monthly, collector, status, amount_paid, total_paid, ref_no, date_paid, category, activation_date, disconnection_date, reactivation_date, last_modified_on, churn) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                values = (row['account_no'], row['subscriber'], row['address'], row['zone'], row['services'], row['monthly'], row['collector'], row['status'], row['amount_paid'], row['total_paid'], row['ref_no'], row['date_paid'], row['category'], row['activation_date'], row['disconnection_date'], row['reactivation_date'], row['last_modified_on'], row['churn'])
 
                cur.execute(sql, values)
                #cur.execute("SELECT * FROM data")
                #cur.copy_from(readcsv, 'data', sep=',')
                conn.commit()
            flash("CSV File Added Successfully")
            return redirect(url_for('auth.custman'))

        return redirect(url_for('auth.custman'))
        
@auth.route('/customer-management/update/<id>', methods = ['GET', 'POST'])
@login_required
def update(id):
    if current_user.cname == "Kalibo Cable":
        if request.method == 'POST':
            datas = Data.query.get(request.form.get('id'))
            datas.services = request.form['services']
            datas.monthly = request.form['monthly']
            datas.amount_paid = request.form['amount_paid']
            datas.date_paid = request.form['date_paid']
            datas.category = request.form['category']
            datas.disconnection_date = request.form['disconnection_date']
            datas.reactivation_date = request.form['reactivation_date']
            
            if datas.disconnection_date == "" and datas.reactivation_date == "":
                datas.disconnection_date = None
                datas.reactivation_date = None
            elif datas.disconnection_date != "" and datas.reactivation_date == "":
                datas.reactivation_date = None
                datas.disconnection_date = request.form['disconnection_date']
            elif datas.disconnection_date != "" and datas.reactivation_date != "":
                datas.reactivation_date = request.form['reactivation_date']
                datas.disconnection_date = request.form['disconnection_date']
                
            if datas.activation_date != None and datas.disconnection_date == None:
                datas.status = "Active"
            elif datas.activation_date != None and datas.disconnection_date != None:
                datas.status = "Disconnected"
            elif datas.activation_date != None and datas.disconnection_date != None and datas.reactivation_date != None:
                datas.status = "Disconnected"

            if datas.disconnection_date == None:
                datas.churn = 0
            else:
                datas.churn = 1
            
            datas.total_paid = float(datas.total_paid) + float(datas.amount_paid)
            datas.last_modified_on = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            db.session.commit()
            print(datas.amount_paid)
            flash("Customer Record Updated Successfully")
     
            return redirect(url_for('auth.custman'))

        return redirect(url_for('auth.custman'))
 
#This route is for deleting our customer in checkbox
@auth.route('/customer-management/delete-selected', methods = ['GET', 'POST'])
@login_required
def deletecheck():
    if current_user.cname == "Kalibo Cable":
        if request.method == "POST":
            for getid in request.form.getlist("mycheckbox"):
                print(getid)
                db.session.query(Data).filter(Data.id ==getid).delete()
            db.session.commit()
            flash("Customer Records Deleted Successfully")
                     
            return redirect(url_for('auth.custman'))
# End of Customer Management    

# User Profile

@auth.route('/user-profile/edit',methods = ['GET', 'POST']) # Edit User Profile
@login_required
def edit():
        if request.method == 'POST':
            if request.files['image_file']:
                picture_file = save_picture(request.files['image_file'])
                current_user.image_file = picture_file
            current_user.fname = request.form['fname']
            current_user.lname = request.form['lname']
            current_user.cp = request.form['cp']
            current_user.address = request.form['address']
            current_user.bday = request.form['bday']
            current_user.about = request.form['about']
            current_user.fb = request.form['fb']
            current_user.ig = request.form['ig']
            current_user.tw = request.form['tw']
            current_user.linkedin = request.form['linkedin']
            db.session.commit()
            flash("User Profile Updated Successfully")
            image_file = url_for('static', filename='images/' + current_user.image_file)
            return render_template("profile.html", user= current_user, image_file = image_file)
        image_file = url_for('static', filename='images/' + current_user.image_file)
        return render_template("edit.html", user= current_user, image_file = image_file)

@auth.route('/user-profile',methods = ['GET', 'POST'])
@login_required
def profile():
    image_file = url_for('static', filename='images/' + current_user.image_file)
    return render_template("profile.html", user= current_user, image_file = image_file)
        
@login_required
def save_picture(form_picture):
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = _ + f_ext
    picture_path = os.path.join(auth.root_path, 'static/images', picture_fn)
    form_picture.save(picture_path)
    output_size = (250, 250)
    i = Image.open(form_picture)
    i.thumbnail(output_size)
    i.save(picture_path)
    return picture_fn

#saving attachments to attachment folder
def save_file(form_file):
    print(form_file.filename)
    _, f_ext = os.path.splitext(form_file.filename)
    file_fn = _ + f_ext
    file_path = os.path.join(auth.root_path, 'static/attachments',file_fn)
    print(file_path,"kkk")
    form_file.save(file_path)
    print (file_fn, 'xxx')
    return file_path

#saving attachments to file folder
def save_import(form_file):
    print(form_file.filename)
    _, f_ext = os.path.splitext(form_file.filename)
    file_fn = _ + f_ext
    file_path = os.path.join(auth.root_path, 'static/file',file_fn)
    print(file_path,"kkk")
    form_file.save(file_path)
    print (file_fn, 'xxx')
    return file_fn
    
#send email
@auth.route('/email-marketing', methods = ['GET','POST'])
@login_required
def send():
    tasks = Task.query.all()
    checker  =  Task.query.count()
    recepients = convert(tasks)
    EMAIL_ADDRESS = 'horizonfeua@gmail.com'
    EMAIL_PASSWORD = 'sleepdep2022'
    if request.method == "POST":
        if checker == 0:
            flash('Please specify at least one recipient.', category="error")
            return redirect(url_for('auth.send'))
        else:
            x = [] 
            if request.files['attfile']:
                    file_attachments =''
                    form_file = save_file(request.files['attfile'])
                    print (form_file,"asdsa")
                    file_attachments = form_file
                    print (file_attachments)
                    x.append(file_attachments)
            contact =  recepients
            msg = MIMEMultipart()
            msg['Subject'] = request.form['subject']
            msg['To'] = ", ".join(recepients) 
            emailMsg=""
            emailMsg = request.form['message']
            msg.attach(MIMEText(emailMsg,'plain'))
            for attachment in x:
                print (attachment)
                content_type, encoding= mimetypes.guess_type(attachment)
                main_type,sub_type = content_type.split('/',1)
                file_name = os.path.basename(attachment)
                f = open(attachment,'rb')
                myFile = MIMEBase(main_type, sub_type)
                myFile.set_payload(f.read())
                myFile.add_header('Content-Disposition', 'attachment', filename=file_name)
                encoders.encode_base64(myFile)
                f.close()
                msg.attach(myFile)
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                smtp.send_message(msg)
                Task.query.delete()
                db.session.commit()
                flash('Message successfully sent.')
                return redirect(url_for('auth.send'))
    image_file = url_for('static', filename='images/' + current_user.image_file)
    return render_template('email-marketing.html', tasks=tasks, recepients = recepients, user= current_user, image_file = image_file)
  
 
#convert query into list of string
def convert(any):
    x = any
    new_strings=[]
    email_strings = []
    for stringx in x:
        new_string = str(stringx)
        new_strings.append(new_string)
    for string in new_strings:
        first_string = string.replace("<Content ","")
        second_string = first_string.replace(">","")
        email_strings.append(second_string)
    print(email_strings)
    return(email_strings)

#add email
@auth.route('/add-email', methods=['POST'])
def add_task():
    content = request.form['content']
    if not content:
        return 'Error'
    task = Task(content)
    db.session.add(task)
    db.session.commit()
    return redirect(url_for('auth.send'))
    
#delete email
@auth.route('/delete/<int:task_id>')
def delete_task(task_id):
    task = Task.query.get(task_id)
    if not task:
        return redirect(url_for('auth.send'))
    db.session.delete(task)
    db.session.commit()
    return redirect(url_for('auth.send'))
    
# Strategies
@auth.route('/strategies', methods=["GET", "POST"])
@login_required
def strat():
    if current_user.explore == "sample":
        statc = Samplestrategies \
            .query \
            .filter(Samplestrategies.status == "complete").count()

        statss = Samplestrategies \
            .query \
            .filter(Samplestrategies.status == "ongoing").count()

        all_data = Samplestrategies.query.all() 
        image_file = url_for('static', filename='images/' + current_user.image_file)
        return render_template("strategies.html", user= current_user, samplestrat=all_data, statss=statss, statc=statc, image_file = image_file)
		
    if current_user.cname == "Kalibo Cable" :
        statc = Strategies \
            .query \
            .filter(Strategies.status == "complete").count()

        statss = Strategies \
            .query \
            .filter(Strategies.status == "ongoing").count()

        all_data = Strategies.query.all() 

        image_file = url_for('static', filename='images/' + current_user.image_file)
        return render_template("strategies.html", user= current_user, strategiess=all_data, statss=statss, statc=statc, image_file = image_file)
            
@auth.route('/strategies/insert', methods = ['POST'])
@login_required
def newstrat():
    if current_user.explore == "sample":
        if request.method == 'POST':
            name = request.form['name']
            act = request.form['act']
            platform = request.form['platform']
            startdate = request.form['startdate']
            enddate = request.form['enddate']
            #status = request.form['status']
            description = request.form['description']

            dates = date.today()
            start = datetime.strptime(startdate, "%Y-%m-%d")
            startdate = start.date()
            endd = datetime.strptime(enddate, "%Y-%m-%d")
            enddate = endd.date()
            end = enddate
            if end == dates or dates > end:
                status = "complete"
            else:
                status = "ongoing"
            print(dates)
            print(end)
            
            row = Samplestrategies.query.count()
            count = Samplestrategies.query.filter(Samplestrategies.id >= row).count()
            if count >= 1:
                id = row + count
                my_strat = Samplestrategies(id=id)
		
            my_strat = Samplestrategies(name=name, act=act, platform=platform, startdate=startdate, 
                        enddate=enddate, status=status, description=description)
            db.session.add(my_strat)
            db.session.commit() 
            
            flash("Strategy Added Successfully")
            
            return redirect(url_for('auth.strat'))
			
    if current_user.cname == "Kalibo Cable":
        if request.method == 'POST':
            name = request.form['name']
            act = request.form['act']
            platform = request.form['platform']
            startdate = request.form['startdate']
            enddate = request.form['enddate']
            #status = request.form['status']
            description = request.form['description']
            
            dates = date.today()
            start = datetime.strptime(startdate, "%Y-%m-%d")
            startdate = start.date()
            endd = datetime.strptime(enddate, "%Y-%m-%d")
            enddate = endd.date()
            end = enddate
            if end == dates or dates > end:
                status = "complete"
            else:
                status = "ongoing"
            print(dates)
            print(end)
            
            row = Strategies.query.count()
            count = Strategies.query.filter(Strategies.id >= row).count()
            if count >= 1:
                id = row + count
                my_strat = Strategies(id=id)
                
            my_strat = Strategies(name=name, act=act, platform=platform, startdate=startdate, 
                        enddate=enddate, status=status, description=description)
            db.session.add(my_strat)
            db.session.commit() 
            
            flash("Strategy Added Successfully")
            
            return redirect(url_for('auth.strat'))
            
@auth.route('/strategies/update/<id>', methods = ['GET', 'POST'])
@login_required
def updatestrat(id):
    if current_user.explore == "sample":
        if request.method == 'POST':
            my_strat = Samplestrategies.query.get(request.form.get('id'))
            my_strat.name = request.form['name']
            my_strat.act = request.form['act']
            my_strat.platform = request.form['platform']
            my_strat.startdate = request.form['startdate']
            my_strat.enddate = request.form['enddate']
            my_strat.status = request.form['status']
            my_strat.description = request.form['description']
            
            dates = date.today()
            start = datetime.strptime(my_strat.startdate, "%Y-%m-%d")
            my_strat.startdate = start.date()
            endd = datetime.strptime(my_strat.enddate, "%Y-%m-%d")
            my_strat.enddate = endd.date()
            end = my_strat.enddate
            if end == dates or dates > end:
                my_strat.status = "complete"
            else:
                my_strat.status = "ongoing"
            
            db.session.commit()
            flash("Strategy Updated Successfully", category="notlimit")
            return redirect(url_for('auth.strat'))
			
    if current_user.cname == "Kalibo Cable":
        if request.method == 'POST':
            my_strat = Strategies.query.get(request.form.get('id'))
            my_strat.name = request.form['name']
            my_strat.act = request.form['act']
            my_strat.platform = request.form['platform']
            my_strat.startdate = request.form['startdate']
            my_strat.enddate = request.form['enddate']
            my_strat.description = request.form['description']
            
            dates = date.today()
            start = datetime.strptime(my_strat.startdate, "%Y-%m-%d")
            my_strat.startdate = start.date()
            endd = datetime.strptime(my_strat.enddate, "%Y-%m-%d")
            my_strat.enddate = endd.date()
            end = my_strat.enddate
            if end == dates or dates > end:
                my_strat.status = "complete"
            else:
                my_strat.status = "ongoing"

            db.session.commit()
            flash("Strategy Updated Successfully", category="notlimit")
            return redirect(url_for('auth.strat')) 

#This route is for deleting our strat
@auth.route('/strategies/delete/<id>/', methods = ['GET', 'POST'])
@login_required
def deletestrat(id):
    if current_user.explore == "sample":
        my_data = Samplestrategies.query.get(id)
        db.session.delete(my_data)
        db.session.commit()
        flash("Strategy Deleted Successfully")
        
        return redirect(url_for('auth.strat'))
		
    if current_user.cname == "Kalibo Cable":
        my_data = Strategies.query.get(id)
        db.session.delete(my_data)
        db.session.commit()
        flash("Strategy Deleted Successfully")
        
        return redirect(url_for('auth.strat'))

#This route is for deleting our strategy in checkbox
@auth.route('/strategies/delete-selected', methods = ['GET', 'POST'])
@login_required
def deletestratcheck():
    if current_user.explore == "sample":
        if request.method == "POST":
            for getid in request.form.getlist("mycheckbox"):
                print(getid)
                db.session.query(Samplestrategies).filter(Samplestrategies.id ==getid).delete()
            db.session.commit()
            flash("Strategy Deleted Successfully")
                     
            return redirect(url_for('auth.strat'))
			
    if current_user.cname == "Kalibo Cable":
        if request.method == "POST":
            for getid in request.form.getlist("mycheckbox"):
                print(getid)
                db.session.query(Strategies).filter(Strategies.id ==getid).delete()
            db.session.commit()
            flash("Strategy Deleted Successfully")
                     
            return redirect(url_for('auth.strat'))
   
# End of Strategies


#send email reset 
def send_reset_email(user):
    print(user)
    msg = EmailMessage()
    EMAIL_ADDRESS = 'horizonfeua@gmail.com'
    EMAIL_PASSWORD = 'sleepdep2022'
    token = user.get_reset_token()
    msg['To'] = [user.email]
    msg['Subject'] = 'Password Reset Request'
    body = f'''
    To reset your password, visit the following link:
    {url_for('auth.reset_token',token=token,_external=True)}
    Disregard this email if you did not make any request
    '''
    msg.set_content(body)
    
    with smtplib.SMTP_SSL('smtp.gmail.com',465) as smtp:
        smtp.login(EMAIL_ADDRESS,EMAIL_PASSWORD)
        smtp.send_message(msg)

#redirect reset request
@auth.route("/sign-in/reset-password",methods=['GET','POST'])
def reset_request():
    if current_user.is_authenticated:
        return redirect(url_for('auth.signin'))
    form = RequestResetForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        send_reset_email(user)
        flash('An email has been sent with instruction to reset your password','info')
        return redirect(url_for('auth.signin'))
    return render_template('reset_request.html',title = 'Forgot your Password?', form = form)
  
  
 #redirect reset password
@auth.route("/sign-in/reset-password/<token>",methods=['GET','POST'])
def reset_token(token):
    if current_user.is_authenticated:
        return redirect(url_for('views.home'))
    user = User.verify_reset_token(token)
    if user is None:
        flash('Invalid or expired token','warning')
        return redirect(url_for('reset_request'))
    form = ResetPasswordForm()
    if form.validate_on_submit():
        user.password = form.password.data
        db.session.commit()
        flash('Your password has been updated! You are now able to log in', 'success')
        return redirect(url_for('auth.signin'))
    return render_template('reset_token.html',title = 'Reset Password', form = form)
