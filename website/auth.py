import os
import secrets
import smtplib
import numpy as np
import pandas as pd
import sqlalchemy
from PIL import Image
from flask import Flask
import base64
import mimetypes
from .extensions import db
from flask import Blueprint, render_template, request, flash, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func, create_engine
from .models import User, Data, Strategies, Contact, Sampledata, Samplestrategies
from flask_login import login_user, login_required, logout_user, current_user
from werkzeug.utils import secure_filename
from website import mail
from datetime import datetime
from itsdangerous import URLSafeTimedSerializer
from flask_mail import Mail, Message
from email.message import EmailMessage
from email import encoders
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.message import EmailMessage

auth = Blueprint('auth', __name__)
  
# Landing Page
#about page
@auth.route('/about')
def about():
    '''cnx = create_engine("sqlite:///website/db.db", echo=True)
    connn = cnx.connect()
    df = pd.read_sql_table('strategies', con=cnx)
    print(df)'''
    return render_template("about.html", user= current_user)
    
#pripo page
@auth.route('/privacy-policy') 
def pripo():
    return render_template("privacypolicy.html", user= current_user)

@auth.route('/terms-conditions') #t&c page
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
    
@auth.route('/contact', methods=["GET", "POST"]) #contact page
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
 
        cont = Contact(name=name, email=email, message=message, cuser_id=current_user.id)
        db.session.add(cont)
        db.session.commit()
    
        flash("Inquiry Successfully Sent!")
        
    return render_template("contact.html", user= current_user)
    
@auth.route('/sign-in', methods=["GET", "POST"]) #signin page
def signin():
    if request.method == "POST" :
        email = request.form.get("email")
        password = request.form.get("password")
        
        user = User.query.filter_by(email=email).first()
        if user:
            if user.password == password:
                if user.user_type == "user":
                    if user.email_confirmed == True:
                        login_user(user, remember=True)
                        return redirect(url_for("views.home"))
                    else:flash("Please confirm your account!", category="error")
                else:
                    flash("You do not have an access to this webpage.", category="error")
            else:
                flash("Password Incorrect. Please try again", category="error")
        else:
            flash("Email does not exists.", category="error")
        
    return render_template("signin.html", user= current_user)

@auth.route('/sign-out') #signout page
@login_required
def signout():
    logout_user()
    return redirect(url_for("views.landing"))

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
            flash("Password must contain 8 characters.\nPlease try again!", category="error")
        else:
            new_user = User(fname=fname, lname=lname, uname=uname, email=email, cname=cname, password=password, user_type=user_type)
            db.session.add(new_user)
            db.session.commit()
            send_confirmation_email(new_user.email)
            flash("Thank you for registering!, Please check your email to confirm your account", category="success")
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

@auth.route('/confirm-email')
def ce():
    return render_template("confirmemail.html", user= current_user)
    
@auth.route('/sign-up/confirm/<token>')
def confirm_email(token):
    try:
        confirm_serializer = URLSafeTimedSerializer('asdfghjkl')
        email = confirm_serializer.loads(token, salt='email-confirmation-salt', max_age=3600)
    except:
        flash('The confirmation link is invalid')
        return redirect(url_for('auth.signin'))
        
    user = User.query.filter_by(email=email).first()
    
    if user.email_confirmed:
        flash('Account confirmed')
    else:
        user.email_confirmed = True 
        user.email_confirmed_on = datetime.now()
        db.session.add(user)
        db.session.commit()
        flash('Thank your for confirming email')
    login_user(user, remember=True)   
    return redirect(url_for('views.home'))
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
    if current_user.cname == "Kalibo":
        all_data = Data.query.all()
        image_file = url_for('static', filename='images/' + current_user.image_file)
        return render_template("custman.html", user= current_user, datas=all_data, image_file = image_file)
        all_data = Data.query.all() 
        return render_template("custman.html", user=current_user, datas=all_data)
    else:
        sd = Sampledata \
            .query \
            .join(User) \
            .filter(User.id==current_user.id).count()
        print(sd)
        if request.method == "POST":
            accnt_num = request.form['accnt_num']
            name = request.form['name']
            address = request.form['address']
            services= request.form['services']
            monthly = request.form['monthly']
            collector = request.form['collector']
            sstatus = request.form['sstatus']
            amnt_paid = request.form['amnt_paid']
            ref_num = request.form['ref_num']

        image_file = url_for('static', filename='images/' + current_user.image_file)
        return render_template("scustman.html", user= current_user, sd=sd, image_file = image_file)
        return render_template("scustman.html", user=current_user, sd=sd)

@auth.route('/customer-management/insert', methods = ['POST'])
@login_required
def insert():
    if current_user.cname == "Kalibo":
        if request.method == 'POST':
            accnt_num = request.form['accnt_num']
            name = request.form['name']
            address = request.form['address']
            services= request.form['services']
            monthly = request.form['monthly']
            collector = request.form['collector']
            sstatus = request.form['sstatus']
            amnt_paid = request.form['amnt_paid']
            ref_num = request.form['ref_num']
     
            datas = Data(accnt_num=accnt_num, name=name, address=address, services=services, monthly=monthly
                        , collector=collector, sstatus=sstatus, amnt_paid=amnt_paid, ref_num=ref_num, duser_id=current_user.id)
            db.session.add(datas)
            db.session.commit()
            
            flash("Customer Inserted Successfully")
            
            return redirect(url_for('auth.custman'))
    else:
        sd = Sampledata \
            .query \
            .join(User) \
            .filter(User.id==current_user.id).count()
        print(sd)
        if request.method == 'POST':
            accnt_num = request.form['accnt_num']
            name = request.form['name']
            address = request.form['address']
            services= request.form['services']
            monthly = request.form['monthly']
            collector = request.form['collector']
            sstatus = request.form['sstatus']
            amnt_paid = request.form['amnt_paid']
            ref_num = request.form['ref_num']
            
            if sd <= 1:
                sdatas = Sampledata(accnt_num=accnt_num, name=name, address=address, services=services, monthly=monthly
                            , collector=collector, sstatus=sstatus, amnt_paid=amnt_paid, ref_num=ref_num, sduser_id=current_user.id)
                db.session.add(sdatas)
                db.session.commit()   
                flash("Customer Inserted Successfully", category="notlimit")
            else:
                db.session.commit()
                flash("You have exceed to the number of inputted customer record!", category="limit")
            
            return redirect(url_for('auth.custman'))
            return render_template(sd=sd)

@auth.route('/customer-management/update/<id>', methods = ['GET', 'POST'])
@login_required
def update(id):
    if current_user.cname == "Kalibo":
        if request.method == 'POST':
            datas = Data.query.get(request.form.get('id'))
            datas.accnt_num = request.form['accnt_num']
            datas.name = request.form['name']
            datas.address = request.form['address']
            datas.services= request.form['services']
            datas.monthly = request.form['monthly']
            datas.collector = request.form['collector']
            datas.sstatus = request.form['sstatus']
            datas.amnt_paid = request.form['amnt_paid']
            datas.ref_num = request.form['ref_num']
            
            db.session.commit()
            
            flash("Customer Updated Successfully")
     
            return redirect(url_for('auth.custman'))
    else:
        if request.method == 'POST':
            datas = Sampledata.query.get(request.form.get('id'))
            datas.accnt_num = request.form['accnt_num']
            datas.name = request.form['name']
            datas.address = request.form['address']
            datas.services= request.form['services']
            datas.monthly = request.form['monthly']
            datas.collector = request.form['collector']
            datas.sstatus = request.form['sstatus']
            datas.amnt_paid = request.form['amnt_paid']
            datas.ref_num = request.form['ref_num']
            
            db.session.commit()
            
            flash("Customer Updated Successfully")
     
            return redirect(url_for('auth.custman'))

#This route is for deleting our customer
@auth.route('/customer-management/delete/<id>/', methods = ['GET', 'POST'])
@login_required
def delete(id):
    if current_user.cname == "Kalibo":
        my_data = Data.query.get(id)
        db.session.delete(my_data)
        db.session.commit()
        flash("Customer Deleted Successfully")
     
        return redirect(url_for('auth.custman'))
    else:
        my_data = Sampledata.query.get(id)
        db.session.delete(my_data)
        db.session.commit()
        flash("Customer Deleted Successfully")
     
        return redirect(url_for('auth.custman'))
 
#This route is for deleting our customer in checkbox
@auth.route('/customer-management/delete-selected', methods = ['GET', 'POST'])
@login_required
def deletecheck():
    if current_user.cname == "Kalibo":
        if request.method == "POST":
            for getid in request.form.getlist("mycheckbox"):
                print(getid)
                db.session.query(Data).filter(Data.id ==getid).delete()
            db.session.commit()
            flash("Customer Deleted Successfully")
                     
            return redirect(url_for('auth.custman'))
    else:
        if request.method == "POST":
            for getid in request.form.getlist("mycheckbox"):
                print(getid)
                db.session.query(Sampledata).filter(Sampledata.id ==getid).delete()
            db.session.commit()
            flash("Customer Deleted Successfully")
                     
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
            flash("User Updated Successfully")
            image_file = url_for('static', filename='images/' + current_user.image_file)
            return render_template("profile.html", user= current_user, image_file = image_file)
        image_file = url_for('static', filename='images/' + current_user.image_file)
        return render_template("edit.html", user= current_user, image_file = image_file)

@auth.route('/user-profile',methods = ['GET', 'POST'])
@login_required
def profile():
    image_file = url_for('static', filename='images/' + current_user.image_file)
    return render_template("profile.html", user= current_user, image_file = image_file)
    return render_template("profile.html", user= current_user)
        
        
@login_required
def save_picture(form_picture):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
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
#send email
@auth.route('/email-marketing', methods = ['GET','POST'])
@login_required
def emailmark():
    EMAIL_ADDRESS = '201811294@feualabang.edu.ph'
    EMAIL_PASSWORD = 'hildeguard'
    contacts = ['YourAddress@gmail.com', 'test@example.com']
    if request.method == "POST":
        x = [] 
        if request.files['attfile']:
                file_attachments =''
                form_file = save_file(request.files['attfile'])
                print (form_file,"asdsa")
                file_attachments = form_file
                print (file_attachments)
                x.append(file_attachments)
        msg = MIMEMultipart()
        msg['Subject'] = request.form['subject']
        msg['To'] = request.form['email']
        emailMsg=""
        emailMsg = request.form['message']
        msg.attach(MIMEText(emailMsg,'plain'))
        print (x)
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
            return redirect(url_for('auth.emailmark'))
    return render_template('email-marketing.html')
    
# Strategies
@auth.route('/strategies', methods=["GET", "POST"])
@login_required
def strat():
    if current_user.cname == "Kalibo":
        statc = Strategies \
            .query \
            .filter(Strategies.status == "complete").count()
        print(statc)
        statss = Strategies \
            .query \
            .filter(Strategies.status == "ongoing").count()
        print(statss)
        all_data = Strategies.query.all() 
        image_file = url_for('static', filename='images/' + current_user.image_file)
        return render_template("strategies.html", user= current_user, strategiess=all_data, statss=statss, statc=statc, image_file = image_file)
        all_data = Strategies.query.all()
        return render_template("strategies.html", user=current_user, strategiess=all_data, statss=statss, statc=statc)
    else:
        sd = Samplestrategies \
            .query \
            .join(User) \
            .filter(User.id==current_user.id).count()
        print(sd)
        statc = Samplestrategies \
            .query \
            .join(User) \
            .filter(Samplestrategies.status == "complete") \
            .filter(User.id==current_user.id).count()
        print(statc)
        statss = Samplestrategies \
            .query \
            .join(User) \
            .filter(Samplestrategies.status == "ongoing") \
            .filter(User.id==current_user.id).count()
        print(statss)
        if request.method == 'POST':
            name = request.form['name']
            act = request.form['act']
            platform = request.form['platform']
            startdate = request.form['startdate']
            enddate = request.form['enddate']
            status = request.form['status']
            description = request.form['description']
        
        image_file = url_for('static', filename='images/' + current_user.image_file)
        return render_template("sstrategies.html", user= current_user, statss=statss, statc=statc, image_file = image_file)
        return render_template("sstrategies.html", user=current_user, statc=statc, statss=statss, sd=sd) 
            
@auth.route('/strategies/insert', methods = ['POST'])
@login_required
def newstrat():
    if current_user.cname == "Kalibo":
        if request.method == 'POST':
            name = request.form['name']
            act = request.form['act']
            platform = request.form['platform']
            startdate = request.form['startdate']
            enddate = request.form['enddate']
            status = request.form['status']
            description = request.form['description']
            
            my_strat = Strategies(name=name, act=act, platform=platform, startdate=startdate, 
                        enddate=enddate, status=status, description=description, stratuser_id=current_user.id)
            db.session.add(my_strat)
            db.session.commit() 
            
            flash("Customer Inserted Successfully")
            
            return redirect(url_for('auth.strat'))
    else:
        sd = Samplestrategies \
            .query \
            .join(User) \
            .filter(User.id==current_user.id).count()
        print(sd)
        statc = Samplestrategies \
            .query \
            .join(User) \
            .filter(Samplestrategies.status == "complete") \
            .filter(User.id==current_user.id).count()
        print(statc)
        statss = Samplestrategies \
            .query \
            .join(User) \
            .filter(Samplestrategies.status == "ongoing") \
            .filter(User.id==current_user.id).count()
        print(statss)
        if request.method == 'POST':
            name = request.form['name']
            act = request.form['act']
            platform = request.form['platform']
            startdate = request.form['startdate']
            enddate = request.form['enddate']
            status = request.form['status']
            description = request.form['description']
            
            if sd <= 1:
                my_strat = Samplestrategies(name=name, act=act, platform=platform, startdate=startdate, 
                        enddate=enddate, status=status, description=description, sstratuser_id=current_user.id)
                db.session.add(my_strat)
                db.session.commit()   
                flash("Strategy Inserted Successfully", category="notlimit")
            else:
                db.session.commit()
                flash("You have exceed to the number of inputted strategy record!", category="limit")
            
            return redirect(url_for('auth.strat'))
            return render_template(sd=sd)
            
@auth.route('/strategies/update/<id>', methods = ['GET', 'POST'])
@login_required
def updatestrat(id):
    if current_user.cname == "Kalibo":
        if request.method == 'POST':
            my_strat = Samplestrategies.query.get(request.form.get('id'))
            my_strat.name = request.form['name']
            my_strat.act = request.form['act']
            my_strat.platform = request.form['platform']
            my_strat.startdate = request.form['startdate']
            my_strat.enddate = request.form['enddate']
            my_strat.status = request.form['status']
            my_strat.description = request.form['description']
            
            db.session.commit()
            return redirect(url_for('auth.strat')) 
    else:
        statc = Samplestrategies \
            .query \
            .join(User) \
            .filter(Samplestrategies.status == "complete") \
            .filter(User.id==current_user.id).count()
        print(statc)
        statss = Samplestrategies \
            .query \
            .join(User) \
            .filter(Samplestrategies.status == "ongoing") \
            .filter(User.id==current_user.id).count()
        print(statss)
        if request.method == 'POST':
            my_strat = Samplestrategies.query.get(request.form.get('id'))
            my_strat.name = request.form['name']
            my_strat.act = request.form['act']
            my_strat.platform = request.form['platform']
            my_strat.startdate = request.form['startdate']
            my_strat.enddate = request.form['enddate']
            my_strat.status = request.form['status']
            my_strat.description = request.form['description']
            
            db.session.commit()
            return redirect(url_for('auth.strat'))
 

#This route is for deleting our strat
@auth.route('/strategies/delete/<id>/', methods = ['GET', 'POST'])
@login_required
def deletestrat(id):
    if current_user.cname == "Kalibo":
        my_data = Strategies.query.get(id)
        db.session.delete(my_data)
        db.session.commit()
        flash("Strategy Deleted Successfully")
        
        return redirect(url_for('auth.strat'))
    else:
        my_data = Samplestrategies.query.get(id)
        db.session.delete(my_data)
        db.session.commit()
        flash("Strategy Deleted Successfully")
        
        return redirect(url_for('auth.strat'))

#This route is for deleting our strategy in checkbox
@auth.route('/strategies/delete-selected', methods = ['GET', 'POST'])
@login_required
def deletestratcheck():
    if current_user.cname == "Kalibo":
        if request.method == "POST":
            for getid in request.form.getlist("mycheckbox"):
                print(getid)
                db.session.query(Strategies).filter(Strategies.id ==getid).delete()
            db.session.commit()
            flash("Strategy Deleted Successfully")
                     
            return redirect(url_for('auth.strat'))
    else:
        if request.method == "POST":
            for getid in request.form.getlist("mycheckbox"):
                print(getid)
                db.session.query(Samplestrategies).filter(Samplestrategies.id ==getid).delete()
            db.session.commit()
            flash("Strategy Deleted Successfully")
                     
            return redirect(url_for('auth.strat'))
    
# End of Strategies
