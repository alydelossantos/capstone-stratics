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
from .models import User, Data, Strategies, Contact, Sampledata, Otherdata, Otherstrategies
from flask_login import login_user, login_required, logout_user, current_user

# Plotly Libraries
import json
import plotly
import plotly.express as px

# Data Preprocessing
import matplotlib.pyplot as plt
import scipy as sp
import scipy._lib

from .extensions import db
views = Blueprint('views', __name__)

@views.route('/home', methods=["GET", "POST"])
@login_required
def home():
    if current_user.explore == "sample":
        current_user.dname = "Sample Dataset"
        db.session.commit()
        cnx = create_engine("postgresql://jzyiaknneqredi:b3f16c49a8b520b2d627ba916908f41bc0a507f7cac2efcb23fa3a8947d76fa8@ec2-35-169-43-5.compute-1.amazonaws.com:5432/dc0chgkng9ougq", echo=True)
        conn = cnx.connect()
        df = pd.read_sql_table('sampledata', con=cnx)
        
    elif current_user.explore == "customer":
        current_user.dname = "Customer Dataset"
        db.session.commit()
        cnx = create_engine("postgresql://jzyiaknneqredi:b3f16c49a8b520b2d627ba916908f41bc0a507f7cac2efcb23fa3a8947d76fa8@ec2-35-169-43-5.compute-1.amazonaws.com:5432/dc0chgkng9ougq", echo=True)
        conn = cnx.connect()
        df = pd.read_sql_table('sampledata', con=cnx)
       
    else:
        current_user.dname = "Enter Dashboard Name"
        db.session.commit()
        cnx = create_engine("postgresql://jzyiaknneqredi:b3f16c49a8b520b2d627ba916908f41bc0a507f7cac2efcb23fa3a8947d76fa8@ec2-35-169-43-5.compute-1.amazonaws.com:5432/dc0chgkng9ougq", echo=True)
        conn = cnx.connect()
        df = pd.read_sql_table('sampledata', con=cnx)
      
    image_file = url_for('static', filename='images/' + current_user.image_file)
    return render_template("home.html", user= current_user, image_file=image_file, graph1JSON=graph1JSON, 
    graph2JSON=graph2JSON, 
    graph3JSON=graph3JSON,
    graph4JSON=graph4JSON,) 


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


