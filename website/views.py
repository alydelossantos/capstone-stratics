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

from .extensions import db
views = Blueprint('views', __name__)

kfull = "Kalibo Cable Television Network, Inc."
knoinc = "Kalibo Cable Television Network"
knonet = "Kalibo Cable Television Network"
knotel = "Kalibo Cable"
knocable = "Kalibo"
abbrenoinc = "KCTN"

@views.route('/home', methods=["GET", "POST"])
@login_required
def home():
    if current_user.explore == "sample":
        cnx = create_engine("postgresql://jzyiaknneqredi:b3f16c49a8b520b2d627ba916908f41bc0a507f7cac2efcb23fa3a8947d76fa8@ec2-35-169-43-5.compute-1.amazonaws.com:5432/dc0chgkng9ougq", echo=True)
        conn = cnx.connect()
        df = pd.read_sql_table('sampledata', con=cnx)

        # Label Encoder
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()

        # Label encoding for columns with 2 or less unique

        le_count = 0
        for col in df.columns[1:]:
            if df[col].dtype == 'object':
                if len(list(df[col].unique())) <=2:
                    le.fit(df[col])
                    df[col] = le.transform(df[col])
                    le_count +=1
        # print('{} columns label encoded'.format(le_count))

        # df2 = df[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges']]
        # fig = plt.figure(figsize=(15, 10))

        # for i in range(df2.shape[1]):
        #     plt.subplot(6, 3, i+1)
        #     f=plt.gca()
        #     f.set_title(df2.columns.values[i])

        # vals = np.size(df2.iloc[:, i].unique())
        # if vals >= 100:
        #     vals = 100

        # plt.hist(df2.iloc[:, i], bins=vals, color = '#f39519')
        # plt.tight_layout()

        # Pie Chart
        from plotly.offline import init_notebook_mode,iplot
        import plotly.graph_objects as go
        import cufflinks as cf
        init_notebook_mode(connected=True)

        #Gender Distribution
        lab = df["gender"].value_counts().keys().tolist()
        #values
        val = df["gender"].value_counts().values.tolist()
        trace = go.Pie(labels=lab, 
                        values=val, 
                        marker=dict(colors=['#9da4d8', '#ed7071']),
                        hole = 0.4,
                        # Seting values to 
                        hoverinfo="value")
        data = [trace]

        layout = go.Layout(dict(title="Gender Distribution",
                    plot_bgcolor = "white",
                    paper_bgcolor = "white",))
        fig1gender = go.Figure(data = data,layout = layout)
        graph1JSON = json.dumps(fig1gender, cls=plotly.utils.PlotlyJSONEncoder)

        #Senior Citizen Distribution
        lab = df["SeniorCitizen"].value_counts().keys().tolist()
        #values
        val = df["SeniorCitizen"].value_counts().values.tolist()
        trace = go.Pie(labels=lab, 
                        values=val, 
                        marker=dict(colors=['#ffa14a', '#ed7071']),
                        hole = 0.4,
                        # Seting values to 
                        hoverinfo="value")
        data = [trace]

        layout = go.Layout(dict(title="% of Senior Citizens",
                    plot_bgcolor = "white",
                    paper_bgcolor = "white",))
        fig2senior = go.Figure(data = data,layout = layout)
        graph2JSON = json.dumps(fig2senior, cls=plotly.utils.PlotlyJSONEncoder)


        # Histogram - Dependents
        # defining data
        trace = go.Histogram(x=df['Dependents'],nbinsx=3,
                        marker = dict(color = '#ed7071'))
        data = [trace]
        # defining layout
        layout = go.Layout(title="Dependents Distribution")
        # defining figure and plotting
        fig3dependents = go.Figure(data = data,layout = layout)
        graph3JSON = json.dumps(fig3dependents, cls=plotly.utils.PlotlyJSONEncoder)


        # Histogram - Partner 
        # defining data
        trace = go.Histogram(
            x=df['Partner'],
            nbinsx=3,
            marker = dict(color = '#9da4d8')
            )
        data = [trace]
        # defining layout
        layout = go.Layout(title="Partner Distribution")
        # defining figure and plotting
        fig4partner = go.Figure(data = data,layout = layout)
        graph4JSON = json.dumps(fig4partner, cls=plotly.utils.PlotlyJSONEncoder)


        # Histogram - Device Protection
        # defining data
        trace = go.Histogram(x=df['DeviceProtection'],nbinsx=3,marker = dict(color = '#6D9886'))
        data = [trace]
        # defining layout
        layout = go.Layout(title="Device Protection Distribution")
        # defining figure and plotting
        fig5device = go.Figure(data = data,layout = layout)
        graph5JSON = json.dumps(fig5device, cls=plotly.utils.PlotlyJSONEncoder)


        # Histogram - Multiple Lines
        # defining data
        trace = go.Histogram(x=df['MultipleLines'],nbinsx=3,marker = dict(color = '#6D9886'))
        data = [trace]
        # defining layout
        layout = go.Layout(title="Multiple Lines Distribution")
        # defining figure and plotting
        fig6multiple = go.Figure(data = data,layout = layout)
        graph6JSON = json.dumps(fig6multiple, cls=plotly.utils.PlotlyJSONEncoder)

        # Histogram - Online Backup
        # defining data
        trace = go.Histogram(x=df['OnlineBackup'],nbinsx=3,marker = dict(color = '#6D9886'))
        data = [trace]
        # defining layout
        layout = go.Layout(title="Online Backup Distribution")
        # defining figure and plotting
        fig7backup = go.Figure(data = data,layout = layout)
        graph7JSON = json.dumps(fig7backup, cls=plotly.utils.PlotlyJSONEncoder)


        # Histogram - Online Security
        # defining data
        trace = go.Histogram(x=df['OnlineSecurity'],nbinsx=3,marker = dict(color = '#6D9886'))
        data = [trace]
        # defining layout
        layout = go.Layout(title="Online Security Distribution")
        # defining figure and plotting
        fig8security = go.Figure(data = data,layout = layout)
        graph8JSON = json.dumps(fig8security, cls=plotly.utils.PlotlyJSONEncoder)


        # Histogram - Paperless Billing
        # defining data
        trace = go.Histogram(x=df['PaperlessBilling'],nbinsx=3,marker = dict(color = '#6D9886'))
        data = [trace]
        # defining layout
        layout = go.Layout(title="Paperless Billing Distribution")
        # defining figure and plotting
        fig9paperless = go.Figure(data = data,layout = layout)
        graph9JSON = json.dumps(fig9paperless, cls=plotly.utils.PlotlyJSONEncoder)


        # Histogram - Phone Service
        # defining data
        trace = go.Histogram(x=df['PhoneService'],nbinsx=3,marker = dict(color = '#6D9886'))
        data = [trace]
        # defining layout
        layout = go.Layout(title="Phone Service Distribution")
        # defining figure and plotting
        fig10phone = go.Figure(data = data,layout = layout)
        graph10JSON = json.dumps(fig10phone, cls=plotly.utils.PlotlyJSONEncoder)


        # Histogram - Streaming Movies
        # defining data
        trace = go.Histogram(x=df['StreamingMovies'],nbinsx=3,marker = dict(color = '#6D9886'))
        data = [trace]
        # defining layout
        layout = go.Layout(title="Streaming Movies Distribution")
        # defining figure and plotting
        fig11movies = go.Figure(data = data,layout = layout)
        graph11JSON = json.dumps(fig11movies, cls=plotly.utils.PlotlyJSONEncoder)


        # Histogram - Streaming TV
        # defining data
        trace = go.Histogram(x=df['StreamingTV'],nbinsx=3,marker = dict(color = '#6D9886'))
        data = [trace]
        # defining layout
        layout = go.Layout(title="Streaming TV Distribution")
        # defining figure and plotting
        fig12tv = go.Figure(data = data,layout = layout)
        graph12JSON = json.dumps(fig12tv, cls=plotly.utils.PlotlyJSONEncoder)


        # Histogram - Tech Support
        # defining data
        trace = go.Histogram(x=df['TechSupport'],nbinsx=3,marker = dict(color = '#6D9886'))
        data = [trace]
        # defining layout
        layout = go.Layout(title="Technical Support Distribution")
        # defining figure and plotting
        fig13techsupport = go.Figure(data = data,layout = layout)
        graph13JSON = json.dumps(fig13techsupport, cls=plotly.utils.PlotlyJSONEncoder)
        
        image_file = url_for('static', filename='images/' + current_user.image_file)
        return render_template("home.html", user= current_user, image_file=image_file,
        graph1JSON=graph1JSON, 
        graph2JSON=graph2JSON, 
        graph3JSON=graph3JSON,
        graph4JSON=graph4JSON,
        graph5JSON=graph5JSON,
        graph6JSON=graph6JSON,
        graph7JSON=graph7JSON,
        graph8JSON=graph8JSON,
        graph9JSON=graph9JSON,
        graph10JSON=graph10JSON,
        graph11JSON=graph11JSON,
        graph12JSON=graph12JSON,
        graph13JSON=graph13JSON)

    elif current_user.explore == "customer":
        if current_user.cname.lower() == kfull.lower() or current_user.cname.lower() == knoinc.lower() or current_user.cname.lower() == knonet.lower() or current_user.cname.lower() == knotel.lower() or current_user.cname.lower() == knocable.lower() or current_user.cname.lower() == abbrenoinc.lower():
            if db.session.query(Data).count() >=3 :
                cnx = create_engine("postgresql://jzyiaknneqredi:b3f16c49a8b520b2d627ba916908f41bc0a507f7cac2efcb23fa3a8947d76fa8@ec2-35-169-43-5.compute-1.amazonaws.com:5432/dc0chgkng9ougq", echo=True)
                conn = cnx.connect()
                df = pd.read_sql_table('data', con=cnx)

                # independent variable
                X = df.iloc[:,:1].values
                X

                # dependent variable - churn column
                y = df.iloc[:,8]
                y

                # Counts number of null values - resulted that no values are missing.
                null_columns=df.columns[df.isnull().any()]
                df[null_columns].isnull().sum()

                # Splitting Data into Train and Test
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

                # Zscore
                from scipy import stats
                zscore = np.abs(stats.zscore(df['monthly']))

                # zscore values higher than 3 are outliers.
                threshold = 3

                df.corr(method='pearson')

                # Create Pivot Table - compute for sum
                pd.pivot_table(df, index=['address', 'services'], aggfunc = 'sum')

                # Create Pivot Table - compute for mean
                pd.pivot_table(df, index=['address', 'services'], aggfunc = 'mean')    

                # Create Pivot Table - compute for count
                pd.pivot_table(df, index=['address', 'services'], aggfunc = 'count')

                # Pie Chart
                from plotly.offline import init_notebook_mode,iplot
                import plotly.graph_objects as go
                import cufflinks as cf
                init_notebook_mode(connected=True)

                #labels
                lab = df["collector"].value_counts().keys().tolist()
                #values
                val = df["collector"].value_counts().values.tolist()
                trace = go.Pie(labels=lab, 
                                values=val, 
                                marker=dict(colors=['red']), 
                                # Seting values to 
                                hoverinfo="value"
                            )
                data = [trace]

                layout = go.Layout(title="Collector")
                fig1 = go.Figure(data = data,layout = layout)
                fig1.update_traces(hole=.4)
                graph1JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

                # Histogram - Service
                # defining data
                trace = go.Histogram(x=df['services'],nbinsx=40,histnorm='percent')
                data = [trace]
                # defining layout
                layout = go.Layout(title="Service Distribution")
                # defining figure and plotting
                fig2 = go.Figure(data = data,layout = layout)
                graph2JSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

                # Histogram - State
                # defining data
                trace = go.Histogram(x=df['address'],nbinsx=52)
                data = [trace]
                # defining layout
                layout = go.Layout(title="Address")
                # defining figure and plotting
                fig3 = go.Figure(data = data,layout = layout)
                fig3 = go.Figure(data = data,layout = layout)
                graph3JSON = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)

                # Histogram - Churn
                # defining data
                trace = go.Histogram(x=df['sstatus'],nbinsx=3)
                data = [trace]
                # defining layout
                layout = go.Layout(title="Churn Distribution")
                # defining figure and plotting
                fig4 = go.Figure(data = data,layout = layout)
                graph4JSON = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)
                
                current_user.dash = "full"
                db.session.commit()
                image_file = url_for('static', filename='images/' + current_user.image_file)
                return render_template("home.html", user= current_user, image_file=image_file, graph1JSON=graph1JSON, 
                graph2JSON=graph2JSON, 
                graph3JSON=graph3JSON,
                graph4JSON=graph4JSON)
            elif db.session.query(Data).count() < 3 and db.session.query(Data).count() > 1 :
                flash("Records must contain atleast 3 rows.", category="error")
                current_user.dash = "none"
                db.session.add(current_user)
                db.session.commit()
                image_file = url_for('static', filename='images/' + current_user.image_file)
                return render_template("home.html", user= current_user, image_file=image_file)
            elif db.session.query(Data).count() == 0:
                flash("Records must contain atleast 3 rows.", category="error")
                current_user.dash = "none"
                db.session.add(current_user)
                db.session.commit()
                image_file = url_for('static', filename='images/' + current_user.image_file)
                return render_template("home.html", user= current_user, image_file=image_file)
        else:
            if db.session.query(Otherdata).count() >=3 :
                cnx = create_engine("postgresql://jzyiaknneqredi:b3f16c49a8b520b2d627ba916908f41bc0a507f7cac2efcb23fa3a8947d76fa8@ec2-35-169-43-5.compute-1.amazonaws.com:5432/dc0chgkng9ougq", echo=True)
                conn = cnx.connect()
                df = pd.read_sql_table('otherdata', con=cnx)

                # independent variable
                X = df.iloc[:,:1].values
                X

                # dependent variable - churn column
                y = df.iloc[:,8]
                y

                # Counts number of null values - resulted that no values are missing.
                null_columns=df.columns[df.isnull().any()]
                df[null_columns].isnull().sum()

                # Splitting Data into Train and Test
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

                # Zscore
                from scipy import stats
                zscore = np.abs(stats.zscore(df['monthly']))

                # zscore values higher than 3 are outliers.
                threshold = 3

                df.corr(method='pearson')

                # Create Pivot Table - compute for sum
                pd.pivot_table(df, index=['address', 'services'], aggfunc = 'sum')

                # Create Pivot Table - compute for mean
                pd.pivot_table(df, index=['address', 'services'], aggfunc = 'mean')    

                # Create Pivot Table - compute for count
                pd.pivot_table(df, index=['address', 'services'], aggfunc = 'count')

                # Pie Chart
                from plotly.offline import init_notebook_mode,iplot
                import plotly.graph_objects as go
                import cufflinks as cf
                init_notebook_mode(connected=True)

                #labels
                lab = df["collector"].value_counts().keys().tolist()
                #values
                val = df["collector"].value_counts().values.tolist()
                trace = go.Pie(labels=lab, 
                                values=val, 
                                marker=dict(colors=['red']), 
                                # Seting values to 
                                hoverinfo="value"
                            )
                data = [trace]

                layout = go.Layout(title="Collector")
                fig1 = go.Figure(data = data,layout = layout)
                fig1.update_traces(hole=.4)
                graph1JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

                # Histogram - Service
                # defining data
                trace = go.Histogram(x=df['services'],nbinsx=40,histnorm='percent')
                data = [trace]
                # defining layout
                layout = go.Layout(title="Service Distribution")
                # defining figure and plotting
                fig2 = go.Figure(data = data,layout = layout)
                graph2JSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

                # Histogram - State
                # defining data
                trace = go.Histogram(x=df['address'],nbinsx=52)
                data = [trace]
                # defining layout
                layout = go.Layout(title="Address")
                # defining figure and plotting
                fig3 = go.Figure(data = data,layout = layout)
                fig3 = go.Figure(data = data,layout = layout)
                graph3JSON = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)

                # Histogram - Churn
                # defining data
                trace = go.Histogram(x=df['sstatus'],nbinsx=3)
                data = [trace]
                # defining layout
                layout = go.Layout(title="Churn Distribution")
                # defining figure and plotting
                fig4 = go.Figure(data = data,layout = layout)
                fig4 = go.Figure(data = data,layout = layout)
                graph4JSON = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)
        
                current_user.dash = "full"
                db.session.commit()
                image_file = url_for('static', filename='images/' + current_user.image_file)
                return render_template("home.html", user= current_user, image_file=image_file, graph1JSON=graph1JSON, 
                graph2JSON=graph2JSON, 
                graph3JSON=graph3JSON,
                graph4JSON=graph4JSON)
            elif db.session.query(Otherdata).count() < 3 and db.session.query(Data).count() > 1 :
                flash("Records must contain atleast 3 rows.", category="error")
                current_user.dash = "none"
                db.session.add(current_user)
                db.session.commit()
                image_file = url_for('static', filename='images/' + current_user.image_file)
                return render_template("home.html", user= current_user, image_file=image_file)
            elif db.session.query(Otherdata).count() == 0:
                flash("Records must contain atleast 3 rows.", category="error")
                current_user.dash = "none"
                db.session.add(current_user)
                db.session.commit()
                image_file = url_for('static', filename='images/' + current_user.image_file)
                return render_template("home.html", user= current_user, image_file=image_file)
    elif current_user.explore == "empty":

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

