import os
import numpy as np
import pandas as pd
import sqlalchemy
from PIL import Image
from flask import Flask
from .extensions import db
from flask import Blueprint, render_template, request, flash, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func, create_engine
from .models import User, Data, Strategies, Contact, Sampledata, Samplestrategies
from flask_login import login_user, login_required, logout_user, current_user

from .extensions import db
views = Blueprint('views', __name__)

@views.route('/home', methods=["GET", "POST"])
@login_required
def home():
    if current_user.explore == "Sample Dataset":
        current_user.dname = "Sample Dataset"
    elif current_user.explore == "Customer Dataset":
        current_user.dname = "Customer Dataset"
    else:
        current_user.dname = "Enter Dashboard Name"
    print(current_user.dname)
    image_file = url_for('static', filename='images/' + current_user.image_file)
    return render_template("home.html", user= current_user, image_file=image_file)

@login_required
def sampleanalysis():
    return render_template("confirmemail.html")

@views.route('/home/explore-dataset', methods=["GET", "POST"])    
@login_required
def homeexp():
    if request.method == 'POST':
        current_user.explore = request.form['explore']
        db.session.commit()
    print(current_user.explore)
    return redirect(url_for('views.home'))

@views.route('/')
def landing():
    return render_template("landing.html", user= current_user)
