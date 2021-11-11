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
from .models import User, Data, Sampledata, Otherdata, Strategies, Samplestrategies, Otherstrategies, Contact, Task
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
knoc = "Kalibo Cable Television Network Inc."
knop = "Kalibo Cable Television Network, Inc"
knob = "Kalibo Cable Television Network Inc"
knoinc = "Kalibo Cable Television Network"
knonet = "Kalibo Cable Television"
knotel = "Kalibo Cable"
knocable = "Kalibo"
abbrenoinc = "KCTN"


@views.route('/home', methods=["GET", "POST"])
@login_required
def home():
    if current_user.explore == "sample":
        cnx = create_engine("postgresql://ympxkbvvsaslrc:45cc51f6a20ea1519edcb35bd69cfdfda91968a390ef9fb2291fb8f3c020cf58@ec2-54-160-35-196.compute-1.amazonaws.com:5432/dd3k0hhqki80nh", echo=True)
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

        # Pie Chart
        from plotly.offline import init_notebook_mode,iplot
        import plotly.graph_objects as go
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


            # ------------ CHURN ANALYTICS ----------------

        #Churn Distribution
        lab = df["Churn"].value_counts().keys().tolist()
        #values
        val = df["Churn"].value_counts().values.tolist()
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
        cfig1 = go.Figure(data = data,layout = layout)
        graph14JSON = json.dumps(cfig1, cls=plotly.utils.PlotlyJSONEncoder)

        # Churn Rate by Gender
        plot_by_gender = df.groupby('gender').Churn.mean().reset_index()
        plot_data = [
            go.Bar(
                x=plot_by_gender['gender'],
                y=plot_by_gender['Churn'],
            width = [0.8],
                marker = dict(      
                    color=['#ed7071', '#ffa14a']
                )
            )
        ]

        layout=go.Layout(
            xaxis={"type": "category"},
            yaxis={"title": "Churn Rate"},
            title="Churn Rate by Gender",
            plot_bgcolor = 'white',
            paper_bgcolor = 'white',
        )

        cfig2 = go.Figure(data=plot_data, layout=layout)
        graph15JSON = json.dumps(cfig2, cls=plotly.utils.PlotlyJSONEncoder)

        # Churn Rate by Internet Service
        plot_by_payment = df.groupby('InternetService').Churn.mean().reset_index()
        plot_data = [
            go.Bar(
                x=plot_by_payment['InternetService'],
                y=plot_by_payment['Churn'],
            width = [0.8],
                marker = dict(
                    color=['#ed7071','#ffa14a', '#9da4d8', '#21ced2']
                )
            )
        ]

        layout=go.Layout(
            xaxis={"type": "category"},
            yaxis={"title": "Churn Rate"},
            title="Churn Rate by Internet Service",
            plot_bgcolor = 'white',
            paper_bgcolor = 'white',
        )

        cfig3 = go.Figure(data=plot_data, layout=layout)
        graph16JSON = json.dumps(cfig3, cls=plotly.utils.PlotlyJSONEncoder)

        # Churn Rate by Contract Duration
        plot_by_contract = df.groupby('Contract').Churn.mean().reset_index()
        plot_data = [
            go.Bar(
                x=plot_by_contract['Contract'],
                y=plot_by_contract['Churn'],
                width = [0.8],
                marker = dict(
                    color=['#ffa14a', '#9da4d8', '#21ced2']
                )
            )
        ]

        cfig4 = go.Figure(data=plot_data, layout=layout)
        graph17JSON = json.dumps(cfig4, cls=plotly.utils.PlotlyJSONEncoder)

        # Churn Rate by Payment Method
        plot_by_payment = df.groupby('PaymentMethod').Churn.mean().reset_index()
        plot_data = [
            go.Bar(
                x=plot_by_payment['PaymentMethod'],
                y=plot_by_payment['Churn'],
                width = [0.8],
                marker = dict(
                    color=['#ed7071','#ffa14a', '#9da4d8', '#21ced2']
                )
            )
        ]

        layout=go.Layout(
            xaxis={"type": "category"},
            yaxis={"title": "Churn Rate"},
            title="Churn Rate by Payment Method",
            plot_bgcolor = 'white',
            paper_bgcolor = 'white',
        )

        cfig5 = go.Figure(data=plot_data, layout=layout)
        graph18JSON = json.dumps(cfig5, cls=plotly.utils.PlotlyJSONEncoder)

        # Relationship of Tenure and Churn Rate
        plot_by_tenure = df.groupby('tenure').Churn.mean().reset_index()
        plot_data = [
            go.Scatter(
                x=plot_by_tenure['tenure'],
                y=plot_by_tenure['Churn'],
                mode = "markers",
                name = "Low",
                marker = dict(
                    size = 5,
                    line = dict(width=0.8),
                    color='green'
                ),
            )
        ]

        layout=go.Layout(
            yaxis={"title": "Churn Rate"},
            xaxis={"title": "Tenure"},
            title="Churn Rate and Tenure Relationship",
            plot_bgcolor = 'white',
            paper_bgcolor = 'white',
        )

        cfig6 = go.Figure(data=plot_data, layout=layout)
        graph19JSON = json.dumps(cfig6, cls=plotly.utils.PlotlyJSONEncoder)

        # ------------ End of Churn Analytics ------------

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
        graph13JSON=graph13JSON,
         # For Churn
        graph14JSON=graph14JSON,
        graph15JSON=graph15JSON,
        graph16JSON=graph16JSON,
        graph17JSON=graph17JSON,
        graph18JSON=graph18JSON,
        graph19JSON=graph19JSON
        )
    elif current_user.explore == "empty":

        image_file = url_for('static', filename='images/' + current_user.image_file)
        return render_template("home.html", user= current_user, image_file=image_file)
    elif current_user.explore == "customer":
        if current_user.cname.lower() == kfull.lower() or current_user.cname.lower() == knoinc.lower() or current_user.cname.lower() == knonet.lower() or current_user.cname.lower() == knotel.lower() or current_user.cname.lower() == knocable.lower() or current_user.cname.lower() == abbrenoinc.lower():
            active = Data \
                .query \
                .filter(Data.status == "Active").count()
            disconnected = Data \
                .query \
                .filter(Data.status == "Disconnected").count()
            if db.session.query(Data).count() >=3 :
                cnx = create_engine("postgresql://ympxkbvvsaslrc:45cc51f6a20ea1519edcb35bd69cfdfda91968a390ef9fb2291fb8f3c020cf58@ec2-54-160-35-196.compute-1.amazonaws.com:5432/dd3k0hhqki80nh", echo=True)
                conn = cnx.connect()
                df = pd.read_sql_table('data', con=cnx)

                    # Check for missing values
                df.isna().any()
                # Fill missing values with NaN
                df.fillna('Null')

                # Convert dates to datetype
                df.activation_date = pd.to_datetime(df.activation_date)
                df.disconnection_date = pd.to_datetime(df.disconnection_date)
                df.reactivation_date = pd.to_datetime(df.reactivation_date)
                df.date_paid = pd.to_datetime(df.date_paid)

                df['disconnection_date'] = df['disconnection_date'].dt.strftime('%m-%d-%Y')
                df['reactivation_date'] = df['reactivation_date'].dt.strftime('%m-%d-%Y')
                df['activation_date'] = df['activation_date'].dt.strftime('%m-%d-%Y')
                df['date_paid'] = df['date_paid'].dt.strftime('%m-%d-%Y')


                # ------ Dashboard for Kalibo DS Sales --------

                # Pie Chart
                from plotly.offline import init_notebook_mode,iplot
                import plotly.graph_objects as go
                init_notebook_mode(connected=True)

                #Gender Distribution
                lab = df["zone"].value_counts().keys().tolist()
                #values
                val = df["zone"].value_counts().values.tolist()
                trace = go.Pie(labels=lab, 
                                values=val, 
                                hole = 0.4,
                                # Seting values to 
                                hoverinfo="value")
                data = [trace]

                layout = go.Layout(dict(title="Zone Distribution",
                            plot_bgcolor = "white",
                            paper_bgcolor = "white",))
                fig1zone = go.Figure(data = data,layout = layout)
                graph20JSON = json.dumps(fig1zone, cls=plotly.utils.PlotlyJSONEncoder)

                #Status Distribution
                lab = df["status"].value_counts().keys().tolist()
                #values
                val = df["status"].value_counts().values.tolist()
                trace = go.Pie(labels=lab, 
                                values=val, 
                                marker=dict(colors=['#ffa14a', '#ed7071']),
                                hole = 0.4,
                                # Seting values to 
                                hoverinfo="value")
                data = [trace]

                layout = go.Layout(dict(title="Status Distribution",
                            plot_bgcolor = "white",
                            paper_bgcolor = "white",))
                fig2status = go.Figure(data = data,layout = layout)
                graph21JSON = json.dumps(fig2status, cls=plotly.utils.PlotlyJSONEncoder)

                # Histogram - Services Availed
                # defining data
                trace = go.Histogram(x=df['services'],nbinsx=3,
                                marker = dict(color = '#ed7071'))
                data = [trace]
                # defining layout
                layout = go.Layout(title="Services Availed Distribution")
                # defining figure and plotting
                fig3services = go.Figure(data = data,layout = layout)
                graph22JSON = json.dumps(fig3services, cls=plotly.utils.PlotlyJSONEncoder)

                # Histogram - Category
                # defining data
                trace = go.Histogram(
                    x=df['collector'],
                    nbinsx=3,
                    marker = dict(color = '#9da4d8')
                    )
                data = [trace]
                # defining layout
                layout = go.Layout(title="Collector Distribution")
                # defining figure and plotting
                fig4collector = go.Figure(data = data,layout = layout)
                graph23JSON = json.dumps(fig4collector, cls=plotly.utils.PlotlyJSONEncoder)

                # Histogram - Category
                # defining data
                trace = go.Histogram(
                    x=df['category'],
                    nbinsx=3,
                    marker = dict(color = '#9da4d8')
                    )
                data = [trace]
                # defining layout
                layout = go.Layout(title="Category Distribution")
                # defining figure and plotting
                fig5category = go.Figure(data = data,layout = layout)
                graph24JSON = json.dumps(fig5category, cls=plotly.utils.PlotlyJSONEncoder)

                # ------ End for Kalibo DS Sales -----

                 # ------ Kalibo DS Sales Churn  --------

                #Churn
                lab = df["churn"].value_counts().keys().tolist()
                #values
                val = df["churn"].value_counts().values.tolist()
                trace = go.Pie(labels=lab, 
                                values=val, 
                                hole = 0.4,
                                # Seting values to 
                                hoverinfo="value")
                data = [trace]

                layout = go.Layout(dict(title="Churn",
                            plot_bgcolor = "white",
                            paper_bgcolor = "white",))
                fig6churn = go.Figure(data = data,layout = layout)
                graph25JSON = json.dumps(fig6churn, cls=plotly.utils.PlotlyJSONEncoder)


                # Churn Rate by Services
                plot_by_gender = df.groupby('services').churn.mean().reset_index()
                plot_data = [
                    go.Bar(
                        x=plot_by_gender['services'],
                        y=plot_by_gender['churn'],
                    width = [0.8],
                            marker = dict(
                            color=['#ed7071','#ffa14a', '#9da4d8', '#21ced2']
                        )
                    )
                ]

                layout=go.Layout(
                    xaxis={"type": "category"},
                    yaxis={"title": "Churn Rate"},
                    title="Churn Rate by Services",
                    plot_bgcolor = 'white',
                    paper_bgcolor = 'white',
                )

                fig7services = go.Figure(data=plot_data, layout=layout)
                graph26JSON = json.dumps(fig7services, cls=plotly.utils.PlotlyJSONEncoder)


                # Churn Rate by Zone
                plot_by_payment = df.groupby('zone').churn.mean().reset_index()
                plot_data = [
                    go.Bar(
                        x=plot_by_payment['zone'],
                        y=plot_by_payment['churn'],
                    width = [0.8],
                        marker = dict(
                            color=['#ed7071','#ffa14a', '#9da4d8', '#21ced2']
                        )
                    )
                ]

                layout=go.Layout(
                    xaxis={"type": "category"},
                    yaxis={"title": "Churn Rate"},
                    title="Churn Rate by Zone",
                    plot_bgcolor = 'white',
                    paper_bgcolor = 'white',
                )

                fig8zone = go.Figure(data=plot_data, layout=layout)
                graph27JSON = json.dumps(fig8zone, cls=plotly.utils.PlotlyJSONEncoder)

                # Churn Rate by Category
                plot_by_contract = df.groupby('category').churn.mean().reset_index()
                plot_data = [
                    go.Bar(
                        x=plot_by_contract['category'],
                        y=plot_by_contract['churn'],
                        width = [0.8],
                        marker = dict(
                            color=['#ffa14a', '#9da4d8', '#21ced2']
                        )
                    )
                ]

                layout=go.Layout(
                    xaxis={"type": "category"},
                    yaxis={"title": "Churn Rate"},
                    title="Churn Rate by Category",
                    plot_bgcolor = 'white',
                    paper_bgcolor = 'white',
                )

                fig9contract = go.Figure(data=plot_data, layout=layout)
                graph28JSON = json.dumps(fig9contract, cls=plotly.utils.PlotlyJSONEncoder)


                current_user.dash = "full"
                db.session.add(current_user)
                db.session.commit()
                image_file = url_for('static', filename='images/' + current_user.image_file)
                return render_template("home.html", user= current_user, image_file=image_file,
                    # For Kalibo Sales
                    graph20JSON=graph20JSON,
                    graph21JSON=graph21JSON,
                    graph22JSON=graph22JSON,
                    graph23JSON=graph23JSON,
                    graph24JSON=graph24JSON,
                     # For Kalibo Churn
                    graph25JSON=graph25JSON,
                    graph26JSON=graph26JSON,
                    graph27SON=graph27JSON,
                    graph28JSON=graph28JSON,active=active, disconnected=disconnected
                    )
            elif db.session.query(Data).count() < 3 and db.session.query(Data).count() >= 1 :
                flash("Records must contain atleast 3 rows.", category="error")
                current_user.dash = "none"
                db.session.add(current_user)
                db.session.commit()
                image_file = url_for('static', filename='images/' + current_user.image_file)
                return render_template("home.html", user= current_user, image_file=image_file)
            elif db.session.query(Data).count() < 1:
                flash("Add records in Customer Management.", category="error")
                current_user.dash = "none"
                db.session.add(current_user)
                db.session.commit()
                image_file = url_for('static', filename='images/' + current_user.image_file)
                return render_template("home.html", user= current_user, image_file=image_file)
        else:
            #if db.session.query(Otherdata).join(User).filter(User.id == current_user.id).count() >=3 :
            cnx = create_engine("postgresql://ympxkbvvsaslrc:45cc51f6a20ea1519edcb35bd69cfdfda91968a390ef9fb2291fb8f3c020cf58@ec2-54-160-35-196.compute-1.amazonaws.com:5432/dd3k0hhqki80nh", echo=True)
            conn = cnx.connect()
            df = pd.read_sql_table('otherdata', con=cnx)
            dataf = df.loc[df['odata_id'] == current_user.id]
            row_count = dataf.index
            rc = len(row_count)
            if rc >= 3:
                # ------ Distribution -------

                #Gender Distribution
                lab = dataf["gender"].value_counts().keys().tolist()
                #values
                val = dataf["gender"].value_counts().values.tolist()
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
                fig1 = go.Figure(data = data,layout = layout)
                graph1JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

                #Province Distribution
                lab = dataf["province"].value_counts().keys().tolist()
                #values
                val = dataf["province"].value_counts().values.tolist()
                trace = go.Pie(labels=lab, 
                                values=val, 
                                marker=dict(colors=['#ffa14a', '#ed7071']),
                                hole = 0.4,
                                # Seting values to 
                                hoverinfo="value")
                data = [trace]

                layout = go.Layout(dict(title="Province Distribution",
                            plot_bgcolor = "white",
                            paper_bgcolor = "white",))
                fig2 = go.Figure(data = data,layout = layout)
                graph2JSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

                # Histogram - Services
                # defining data
                trace = go.Histogram(x=dataf['services'],nbinsx=3,
                                marker = dict(color = '#ed7071'))
                data = [trace]
                # defining layout
                layout = go.Layout(title="Services Distribution")
                # defining figure and plotting
                fig3 = go.Figure(data = data,layout = layout)
                graph3JSON = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)

                # Histogram - Category 
                # defining data
                trace = go.Histogram(
                    x=dataf['category'],
                    nbinsx=3,
                    marker = dict(color = '#9da4d8')
                    )
                data = [trace]
                # defining layout
                layout = go.Layout(title="Category Distribution")
                # defining figure and plotting
                fig4 = go.Figure(data = data,layout = layout)
                graph4JSON = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)

                # --------- End of Distribution -----------

                # --------- Start of Churn ----------------

                # Churn Rate by Gender
                plot_by_gender = dataf.groupby('gender').churn.mean().reset_index()
                plot_data = [
                    go.Bar(
                        x=plot_by_gender['gender'],
                        y=plot_by_gender['churn'],
                    width = [0.8],
                        marker = dict(
                            color=['#ed7071', '#ffa14a']
                        )
                    )
                ]

                layout=go.Layout(
                    xaxis={"type": "category"},
                    yaxis={"title": "Churn Rate"},
                    title="Churn Rate by Gender",
                    plot_bgcolor = 'white',
                    paper_bgcolor = 'white',
                )

                fig5 = go.Figure(data=plot_data, layout=layout)
                graph5JSON = json.dumps(fig5, cls=plotly.utils.PlotlyJSONEncoder)

                # Churn Rate by Services
                plot_by_payment = dataf.groupby('services').churn.mean().reset_index()
                plot_data = [
                    go.Bar(
                        x=plot_by_payment['services'],
                        y=plot_by_payment['churn'],
                    width = [0.8],
                        marker = dict(
                            color=['#ed7071','#ffa14a', '#9da4d8', '#21ced2']
                        )
                    )
                ]

                layout=go.Layout(
                    xaxis={"type": "category"},
                    yaxis={"title": "Churn Rate"},
                    title="Churn Rate by Services",
                    plot_bgcolor = 'white',
                    paper_bgcolor = 'white',
                )

                fig6 = go.Figure(data=plot_data, layout=layout)
                graph6JSON = json.dumps(fig6, cls=plotly.utils.PlotlyJSONEncoder)

                # Churn Rate by Province
                plot_by_contract = dataf.groupby('province').churn.mean().reset_index()
                plot_data = [
                    go.Bar(
                        x=plot_by_contract['province'],
                        y=plot_by_contract['churn'],
                        width = [0.8],
                        marker = dict(
                            color=['#ffa14a', '#9da4d8', '#21ced2']
                        )
                    )
                ]

                layout=go.Layout(
                    xaxis={"type": "category"},
                    yaxis={"title": "Churn Rate"},
                    title="Churn Rate by Province",
                    plot_bgcolor = 'white',
                    paper_bgcolor = 'white',
                )

                fig7 = go.Figure(data=plot_data, layout=layout)
                graph7JSON = json.dumps(fig7, cls=plotly.utils.PlotlyJSONEncoder)

                # Churn Rate by Category
                plot_by_payment = dataf.groupby('category').churn.mean().reset_index()
                plot_data = [
                    go.Bar(
                        x=plot_by_payment['category'],
                        y=plot_by_payment['churn'],
                        width = [0.8],
                        marker = dict(
                            color=['#ed7071','#ffa14a', '#9da4d8', '#21ced2']
                        )
                    )
                ]

                layout=go.Layout(
                    xaxis={"type": "category"},
                    yaxis={"title": "Churn Rate"},
                    title="Churn Rate by Category",
                    plot_bgcolor = 'white',
                    paper_bgcolor = 'white',
                )

                fig8 = go.Figure(data=plot_data, layout=layout)
                graph8JSON = json.dumps(fig8, cls=plotly.utils.PlotlyJSONEncoder)
                
                # -------- End of Churn ------------

                current_user.dash = "full"
                db.session.add(current_user)
                db.session.commit()
                image_file = url_for('static', filename='images/' + current_user.image_file)
                return render_template("home.html", user= current_user, image_file=image_file,
                graph1JSON=graph1JSON, 
                graph2JSON=graph2JSON, 
                graph3JSON=graph3JSON,
                graph4JSON=graph4JSON,
                graph5JSON=graph5JSON, 
                graph6JSON=graph6JSON, 
                graph7JSON=graph7JSON,
                graph8JSON=graph8JSON
                )
            elif rc < 3 and rc >= 1:
                flash("Records must contain atleast 3 rows.", category="error")
                current_user.dash = "none"
                db.session.add(current_user)
                db.session.commit()
                image_file = url_for('static', filename='images/' + current_user.image_file)
                return render_template("home.html", user= current_user, image_file=image_file)
            elif rc < 1:
                flash("Add records in Customer Management.", category="error")
                current_user.dash = "none"
                db.session.add(current_user)
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

