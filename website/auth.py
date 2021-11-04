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
  
#signin page   
@auth.route('/', methods=["GET", "POST"]) #signin page
def signin():
    if request.method == "POST" :
        email = request.form.get("email")
        password = request.form.get("password")
        
        user = User.query.filter_by(email=email).first()
        if user:
            if user.password == password:
                if user.user_type == "admin":
                    if user.email_confirmed == True:
                      login_user(user, remember=True)
                      user.user_status = True
                      db.session.add(user)
                      db.session.commit()
                      return redirect(url_for("auth.accounts"))
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
    if current_user:
        current_user.user_status = False
        db.session.commit()
        logout_user()
    return redirect(url_for("auth.signin"))

#SIGNUP PAGE
@auth.route('/sign-up', methods=["GET", "POST"])
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

#SEND CONFIRM EMAIL
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
    return redirect(url_for('auth.accounts'))
    return render_template(user= current_user)
  
# Admin Page

# User Profile
@auth.route('/user-profile/edit',methods = ['GET', 'POST'])
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
          current_user.position = request.form['position']
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
    
#SEND EMAIL FOR INQUIRIES
@auth.route('/inquiries/send-email/<id>', methods = ['GET','POST'])
@login_required
def emailmark(id):
    EMAIL_ADDRESS = 'ksn.080900@gmail.com'
    EMAIL_PASSWORD = 's4noope@cH'
    contacts = ['YourAddress@gmail.com', 'test@example.com']
    if request.method == "POST":
      my_data = Contact.query.get(request.form.get('id'))
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
          db.session.commit()
          return redirect(url_for('auth.inq'))
    
#ACCOUNTS MANAGEMENT
@auth.route('/user-accounts', methods = ['GET', 'POST'])
@login_required
def accounts():
    all_data = User.query.filter_by(user_type="user").all()
    image_file = url_for('static', filename='images/' + current_user.image_file) 
    return render_template("accounts.html", user=current_user, users=all_data, image_file = image_file)

@auth.route('/user-accounts/update/<id>', methods = ['GET', 'POST'])
@login_required
def updateaccnt(id):
    if request.method == 'POST':
        my_data = User.query.get(request.form.get('id'))
        my_data.position = request.form['position']
        my_data.password = request.form['password']
        db.session.commit()
        flash("User Account Updated Successfully")
 
        return redirect(url_for('auth.accounts')) 
 
#This route is for deleting our user accounts
@auth.route('/user-accounts/delete/<id>/', methods = ['GET', 'POST'])
@login_required
def deleteaccnt(id):
    my_data = User.query.get(id)
    db.session.delete(my_data)
    db.session.commit()
    flash("User Account Deleted Successfully")
    
    return redirect(url_for('auth.accounts'))

#This route is for deleting user accnt in checkbox
@auth.route('/user-accounts/delete-selected', methods = ['GET', 'POST'])
@login_required
def deletecheckaccnt():
    if request.method == "POST":
        for getid in request.form.getlist("mycheckbox"):
            print(getid)
            db.session.query(User).filter(User.id==getid).delete()
        db.session.commit()
        flash("User Deleted Successfully")
                 
        return redirect(url_for('auth.accounts'))
# End of Accounts 

#INQUIRIES
@auth.route('/inquiries', methods = ['GET', 'POST'])
@login_required
def inq():
    all_data = Contact.query.all()
    image_file = url_for('static', filename='images/' + current_user.image_file) 
    return render_template("inquiries.html", user=current_user, contacts=all_data, image_file = image_file)
 
#This route is for deleting our inquiries
@auth.route('/inquiries/delete/<id>/', methods = ['GET', 'POST'])
@login_required
def deleteinq(id):
    my_data = Contact.query.get(id)
    db.session.delete(my_data)
    db.session.commit()
    flash("Inquiry Deleted Successfully")
    
    return redirect(url_for('auth.inq'))

#This route is for deleting inquiries in checkbox
@auth.route('/inquiries/delete-selected', methods = ['GET', 'POST'])
@login_required
def deletecheckinq():
    if request.method == "POST":
        for getid in request.form.getlist("mycheckbox"):
            print(getid)
            db.session.query(Contact).filter(Contact.id==getid).delete()
        db.session.commit()
        flash("Selected Inquiries Deleted Successfully")
                 
        return redirect(url_for('auth.inq'))
# End of Inquiries 

