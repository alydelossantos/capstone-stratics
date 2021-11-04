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
        current_user.dname = "Sample Dashboard"
        db.session.commit()
        dashboard()
        image_file = url_for('static', filename='images/' + current_user.image_file)
        return render_template("home.html", user= current_user, image_file=image_file, graph1JSON=graph1JSON, 
        graph2JSON=graph2JSON, 
        graph3JSON=graph3JSON,
        graph4JSON=graph4JSON,)
    elif current_user.explore == "customer":
        current_user.dname = "Edit Dashboard Name"
        db.session.commit()
        dashboard()
        image_file = url_for('static', filename='images/' + current_user.image_file)
        return render_template("home.html", user= current_user, image_file=image_file, graph1JSON=graph1JSON, 
        graph2JSON=graph2JSON, 
        graph3JSON=graph3JSON,
        graph4JSON=graph4JSON,)
    else:
        current_user.dname = "Empty Dashboard"
        db.session.commit()
        image_file = url_for('static', filename='images/' + current_user.image_file)
        return render_template("home.html", user= current_user, image_file=image_file) 
    
    image_file = url_for('static', filename='images/' + current_user.image_file)
    return render_template("home.html", user= current_user, image_file=image_file)

@views.route('/home/dashboard-name/edit', methods=["GET", "POST"])
@login_required
def dashname():
     if request.method == 'POST':
        current_user.dname = request.form['dname']
        db.session.commit()
        return redirect(url_for('views.home'))

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


def dashboard():
    if current_user.explore == "sample":
        cnx = create_engine("postgresql://jzyiaknneqredi:b3f16c49a8b520b2d627ba916908f41bc0a507f7cac2efcb23fa3a8947d76fa8@ec2-35-169-43-5.compute-1.amazonaws.com:5432/dc0chgkng9ougq", echo=True)
        conn = cnx.connect()
        df = pd.read_sql_table('sampledata', con=cnx)
    elif current_user.explore == "customer":
        cnx = create_engine("postgresql://jzyiaknneqredi:b3f16c49a8b520b2d627ba916908f41bc0a507f7cac2efcb23fa3a8947d76fa8@ec2-35-169-43-5.compute-1.amazonaws.com:5432/dc0chgkng9ougq", echo=True)
        conn = cnx.connect()
        df = pd.read_sql_table('sampledata', con=cnx)
    else:
        cnx = create_engine("postgresql://jzyiaknneqredi:b3f16c49a8b520b2d627ba916908f41bc0a507f7cac2efcb23fa3a8947d76fa8@ec2-35-169-43-5.compute-1.amazonaws.com:5432/dc0chgkng9ougq", echo=True)
        conn = cnx.connect()
        df = pd.read_sql_table('sampledata', con=cnx)
        
    # independent variable
    X = df.iloc[:,:-1].values
    X

    # dependent variable - churn column
    y = df.iloc[:,10]
    y

    # Counts number of null values - resulted that no values are missing.
    null_columns=df.columns[df.isnull().any()]
    df[null_columns].isnull().sum()
    
    # Splitting Data into Train and Test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    print("X_train : ",X_train.shape)
    print("X_test : ",X_test.shape)
    print("y_train : ",y_train.shape)
    print("y_test : ",y_test.shape)

    # Outlier Detection
    print(df.shape)
    print(df.columns)

    # Zscore
    from scipy import stats
    zscore = np.abs(stats.zscore(df['MonthlyCharges']))
    print (zscore)

    # zscore values higher than 3 are outliers.
    threshold = 3
    print(np.where(zscore >3))

    df.corr(method='pearson')

    # Create Pivot Table - compute for sum
    pd.pivot_table(df, index=['State', 'InternetService'], aggfunc = 'sum')

    # Create Pivot Table - compute for mean
    pd.pivot_table(df, index=['State', 'InternetService'], aggfunc = 'mean')    
    
    # Create Pivot Table - compute for count
    pd.pivot_table(df, index=['State', 'InternetService'], aggfunc = 'count')

    # Pie Chart
    from plotly.offline import init_notebook_mode,iplot
    import plotly.graph_objects as go
    import cufflinks as cf
    init_notebook_mode(connected=True)

    #labels
    lab = df["gender"].value_counts().keys().tolist()
    #values
    val = df["gender"].value_counts().values.tolist()
    trace = go.Pie(labels=lab, 
                    values=val, 
                    marker=dict(colors=['red']), 
                    # Seting values to 
                    hoverinfo="value"
                )
    data = [trace]

    
    layout = go.Layout(title="Sex Distribution")
    fig1 = go.Figure(data = data,layout = layout)
    fig1.update_traces(hole=.4)
    graph1JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

    # Histogram - Service
    # defining data
    trace = go.Histogram(x=df['InternetService'],nbinsx=40,histnorm='percent')
    data = [trace]
    # defining layout
    layout = go.Layout(title="Service Distribution")
    # defining figure and plotting
    fig2 = go.Figure(data = data,layout = layout)
    graph2JSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

    # Histogram - State
    # defining data
    trace = go.Histogram(x=df['State'],nbinsx=52)
    data = [trace]
    # defining layout
    layout = go.Layout(title="State")
    # defining figure and plotting
    fig3 = go.Figure(data = data,layout = layout)
    fig3 = go.Figure(data = data,layout = layout)
    graph3JSON = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)

    # Histogram - Churn
    # defining data
    trace = go.Histogram(x=df['Churn'],nbinsx=3)
    data = [trace]
    # defining layout
    layout = go.Layout(title="Churn Distribution")
    # defining figure and plotting
    fig4 = go.Figure(data = data,layout = layout)
    fig4 = go.Figure(data = data,layout = layout)
    graph4JSON = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)
