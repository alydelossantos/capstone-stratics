import os
import secrets
import smtplib
import numpy as np
import pandas as pd
import sqlalchemy
from PIL import Image
from flask import Blueprint, render_template, request, flash, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func, create_engine
from .models import User, Data, Strategies, Contact, Sampledata, Samplestrategies
from flask_login import login_user, login_required, logout_user, current_user
from werkzeug.utils import secure_filename
from website import db

auth = Blueprint('auth', __name__)

# Landing Page
@auth.route('/about') #about page
def about():
    '''cnx = create_engine("sqlite:///website/db.db", echo=True)
    connn = cnx.connect()
    df = pd.read_sql_table('strategies', con=cnx)
    print(df)'''
    return render_template("about.html", user= current_user)

@auth.route('/privacy-policy') #pripo page
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
                login_user(user, remember=True)
                return redirect(url_for("views.home"))
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

        user = User.query.filter_by(email=email).first()
        usern = User.query.filter_by(uname=uname).first()
        if user:
            flash("Email already exists.", category="error")
        elif usern:
            flash("Username already exists.", category="error")
        elif len(password) < 8:
            flash("Password must contain 8 characters.\nPlease try again!", category="error")
        else:
            new_user = User(fname=fname, lname=lname, uname=uname, email=email, cname=cname, password=password)
            db.session.add(new_user)
            db.session.commit()
            return redirect(url_for("views.home"))
    return render_template("signup.html", user= current_user)

# Home Page

#SideBar
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
                db.session.commit()
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

# Email Marketing   
@auth.route('/email-marketing')
@login_required
def email():
    image_file = url_for('static', filename='images/' + current_user.image_file)
    return render_template("email.html", user= current_user, image_file = image_file)

@auth.route('/email-marketing', methods = ['GET','POST'])
def send_message():
    if request.method == "POST":
        email = request.form['Email']
        subject = request.form['Subject']
        msg = request.form['Body']
        image_file = url_for('static', filename='images/' + current_user.image_file)
        return render_template("email.html", user= current_user, image_file = image_file)
    
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
    
    
    
    
    
    
    
    
# Account Management

@auth.route('/user-accounts', methods = ['GET', 'POST'])
@login_required
def accounts():
    if request.method == 'POST':

        lname = request.form['name']
        fname = request.form['act']
        address = request.form['platform']
        startdate = request.form['startdate']
        enddate = request.form['enddate']
        status = request.form['status']
        
    return render_template("accounts.html", user=current_user)


@auth.route('/user-accounts/update/<id>', methods = ['GET', 'POST'])
@login_required
def updateaccnt(id):
 
    if request.method == 'POST':
        my_data = User.query.get(request.form.get('id'))
        my_data.lname = request.form['lname']
        my_data.fname = request.form['fname']
        my_data.address = request.form['address']
 
        db.session.commit()
        flash("User Account Updated Successfully")
 
        return render_template("accounts.html", user=current_user)
 

#This route is for deleting our user
@auth.route('/user-accounts/delete/<id>/', methods = ['GET', 'POST'])
@login_required
def deleteaccnt(id):
    my_data = User.query.get(id)
    db.session.delete(my_data)
    db.session.commit()
    flash("User Account Deleted Successfully")
 
    return render_template("accounts.html", user=current_user)

#This route is for deleting our accnt in checkbox
@auth.route('/user-accounts/delete-selected', methods = ['GET', 'POST'])
@login_required
def deletecheckaccnt():
    if request.method == "POST":
        for current_user.id in request.form.getlist("mycheckbox"):
            print(current_user.id)
            db.session.query(User).filter(User.id ==current_user.id).delete()
        db.session.commit()
        flash("User Account Deleted Successfully")
                 
        return render_template("accounts.html", user=current_user)
    return render_template("accounts.html", user=current_user)
    
# End of Accounts  
