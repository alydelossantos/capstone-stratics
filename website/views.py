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
from plotly.offline import init_notebook_mode,iplot
import plotly.graph_objects as go
init_notebook_mode(connected=True)

# Plotly Libraries
import json
import plotly
import plotly.express as px

# Data Preprocessing
import matplotlib.pyplot as plt
import scipy as sp

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


cnx = create_engine("postgresql://ympxkbvvsaslrc:45cc51f6a20ea1519edcb35bd69cfdfda91968a390ef9fb2291fb8f3c020cf58@ec2-54-160-35-196.compute-1.amazonaws.com:5432/dd3k0hhqki80nh", echo=True)
conn = cnx.connect()

@views.route('/home', methods=["GET", "POST"])
@login_required
def home():
    if current_user.explore == "sample":
        total = db.session.query(Sampledata).count()
        avg = 64.76
        ave = 2296.83
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
        # from plotly.offline import init_notebook_mode,iplot
        # import plotly.graph_objects as go
        # init_notebook_mode(connected=True)

        #Gender Distribution
        lab = df["gender"].value_counts().keys().tolist()
        #values
        val = df["gender"].value_counts().values.tolist()
        trace = go.Pie(labels=lab, 
                        values=val, 
                        marker=dict(colors=['#fc636b', '#ffb900', '#6a67ce', '#1aafd0', '#3be8b0']),
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
                        marker=dict(colors=['#fc636b', '#ffb900', '#6a67ce', '#1aafd0', '#3be8b0']),
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
                        marker = dict(color = '#fc636b'))
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
            marker = dict(color = '#ffb900')
            )
        data = [trace]
        # defining layout
        layout = go.Layout(title="Partner Distribution")
        # defining figure and plotting
        fig4partner = go.Figure(data = data,layout = layout)
        graph4JSON = json.dumps(fig4partner, cls=plotly.utils.PlotlyJSONEncoder)


        # Histogram - Device Protection
        # defining data
        trace = go.Histogram(x=df['DeviceProtection'],nbinsx=3,marker = dict(color = '#6a67ce'))
        data = [trace]
        # defining layout
        layout = go.Layout(title="Device Protection Distribution")
        # defining figure and plotting
        fig5device = go.Figure(data = data,layout = layout)
        graph5JSON = json.dumps(fig5device, cls=plotly.utils.PlotlyJSONEncoder)


        # Histogram - Multiple Lines
        # defining data
        trace = go.Histogram(x=df['MultipleLines'],nbinsx=3,marker = dict(color = '#1aafd0'))
        data = [trace]
        # defining layout
        layout = go.Layout(title="Multiple Lines Distribution")
        # defining figure and plotting
        fig6multiple = go.Figure(data = data,layout = layout)
        graph6JSON = json.dumps(fig6multiple, cls=plotly.utils.PlotlyJSONEncoder)

        # Histogram - Online Backup
        # defining data
        trace = go.Histogram(x=df['OnlineBackup'],nbinsx=3,marker = dict(color = '#3be8b0'))
        data = [trace]
        # defining layout
        layout = go.Layout(title="Online Backup Distribution")
        # defining figure and plotting
        fig7backup = go.Figure(data = data,layout = layout)
        graph7JSON = json.dumps(fig7backup, cls=plotly.utils.PlotlyJSONEncoder)


        # Histogram - Online Security
        # defining data
        trace = go.Histogram(x=df['OnlineSecurity'],nbinsx=3,marker = dict(color = '#fc636b'))
        data = [trace]
        # defining layout
        layout = go.Layout(title="Online Security Distribution")
        # defining figure and plotting
        fig8security = go.Figure(data = data,layout = layout)
        graph8JSON = json.dumps(fig8security, cls=plotly.utils.PlotlyJSONEncoder)


        # Histogram - Paperless Billing
        # defining data
        trace = go.Histogram(x=df['PaperlessBilling'],nbinsx=3,marker = dict(color = '#ffb900'))
        data = [trace]
        # defining layout
        layout = go.Layout(title="Paperless Billing Distribution")
        # defining figure and plotting
        fig9paperless = go.Figure(data = data,layout = layout)
        graph9JSON = json.dumps(fig9paperless, cls=plotly.utils.PlotlyJSONEncoder)


        # Histogram - Phone Service
        # defining data
        trace = go.Histogram(x=df['PhoneService'],nbinsx=3,marker = dict(color = '#6a67ce'))
        data = [trace]
        # defining layout
        layout = go.Layout(title="Phone Service Distribution")
        # defining figure and plotting
        fig10phone = go.Figure(data = data,layout = layout)
        graph10JSON = json.dumps(fig10phone, cls=plotly.utils.PlotlyJSONEncoder)


        # Histogram - Streaming Movies
        # defining data
        trace = go.Histogram(x=df['StreamingMovies'],nbinsx=3,marker = dict(color = '#1aafd0'))
        data = [trace]
        # defining layout
        layout = go.Layout(title="Streaming Movies Distribution")
        # defining figure and plotting
        fig11movies = go.Figure(data = data,layout = layout)
        graph11JSON = json.dumps(fig11movies, cls=plotly.utils.PlotlyJSONEncoder)


        # Histogram - Streaming TV
        # defining data
        trace = go.Histogram(x=df['StreamingTV'],nbinsx=3,marker = dict(color = '#3be8b0'))
        data = [trace]
        # defining layout
        layout = go.Layout(title="Streaming TV Distribution")
        # defining figure and plotting
        fig12tv = go.Figure(data = data,layout = layout)
        graph12JSON = json.dumps(fig12tv, cls=plotly.utils.PlotlyJSONEncoder)

        # Histogram - Tech Support
        # defining data
        trace = go.Histogram(x=df['TechSupport'],nbinsx=3,marker = dict(color = '#fc636b'))
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
                        marker=dict(colors=['#fc636b', '#ffb900', '#6a67ce', '#1aafd0', '#3be8b0']),
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
                    color=['#fc636b', '#ffb900', '#6a67ce', '#1aafd0', '#3be8b0']
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
                    color=['#fc636b', '#ffb900', '#6a67ce', '#1aafd0', '#3be8b0']
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
                    color=['#fc636b', '#ffb900', '#6a67ce', '#1aafd0', '#3be8b0']
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
                    color=['#fc636b', '#ffb900', '#6a67ce', '#1aafd0', '#3be8b0']
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
        graph19JSON=graph19JSON,
        total=total,
        avg=avg,ave=ave,
        )

    elif current_user.explore == "customer":
        if current_user.cname == "Kalibo Cable":
            total = db.session.query(Data).count()
            active = Data \
                .query \
                .filter(Data.status == "Active").count()
            disconnected = Data \
                .query \
                .filter(Data.status == "Disconnected").count()
            if db.session.query(Data).count() >=3 :
                kctn = pd.read_sql_table('data', con=cnx)

                # # Pie Chart
                # from plotly.offline import init_notebook_mode,iplot
                # import plotly.graph_objects as go
                # import cufflinks as cf
                # init_notebook_mode(connected=True)

                # Check for missing values
                kctn.isna().any()
                # Fill missing values with NaN
                kctn.fillna('Null')

                # Convert dates to datetype
                kctn.activation_date = pd.to_datetime(kctn.activation_date)
                kctn.disconnection_date = pd.to_datetime(kctn.disconnection_date)
                kctn.reactivation_date = pd.to_datetime(kctn.reactivation_date)
                kctn.date_paid = pd.to_datetime(kctn.date_paid)

                kctn['disconnection_date'] = kctn['disconnection_date'].dt.strftime('%m-%d-%Y')
                kctn['reactivation_date'] = kctn['reactivation_date'].dt.strftime('%m-%d-%Y')
                kctn['activation_date'] = kctn['activation_date'].dt.strftime('%m-%d-%Y')
                kctn['date_paid'] = kctn['date_paid'].dt.strftime('%m-%d-%Y')

                # ------ Dashboard for Kalibo DS Sales --------

                #Zone Distribution
                lab = kctn["zone"].value_counts().keys().tolist()
                #values
                val = kctn["zone"].value_counts().values.tolist()
                trace = go.Pie(labels=lab, 
                                values=val, 
                                hole = 0.4,
                                # Seting values to 
                                hoverinfo="value")
                data = [trace]

                layout = go.Layout(dict(title="Zone Distribution",
                            plot_bgcolor = "white",
                            paper_bgcolor = "white",))
                figs1 = go.Figure(data = data,layout = layout)
                graphs1JSON = json.dumps(figs1, cls=plotly.utils.PlotlyJSONEncoder)

                #labels
                lab = kctn["status"].value_counts().keys().tolist()
                #values
                val = kctn["status"].value_counts().values.tolist()
                trace = go.Pie(labels=lab, 
                                values=val, 
                                marker=dict(colors=['#fc636b', '#ffb900', '#6a67ce', '#1aafd0', '#3be8b0']),
                                hole = 0.4,
                                # Seting values to 
                                hoverinfo="value")
                data = [trace]

                layout = go.Layout(dict(title="Status Distribution",
                            plot_bgcolor = "white",
                            paper_bgcolor = "white",))
                figs2 = go.Figure(data = data,layout = layout)
                graphs2JSON = json.dumps(figs2, cls=plotly.utils.PlotlyJSONEncoder)

                # Histogram - Services Availed
                # defining data
                trace = go.Histogram(x=kctn['services'],nbinsx=3,
                                marker = dict(color = '#ed7071'))
                data = [trace]
                # defining layout
                layout = go.Layout(title="Services Availed Distribution")
                # defining figure and plotting
                figs3 = go.Figure(data = data,layout = layout)
                graphs3JSON = json.dumps(figs3, cls=plotly.utils.PlotlyJSONEncoder)


                # Histogram - Category
                # defining data
                trace = go.Histogram(
                    x=kctn['collector'],
                    nbinsx=3,
                    marker = dict(color = '#ffb900')
                    )
                data = [trace]
                # defining layout
                layout = go.Layout(title="Collector Distribution")
                # defining figure and plotting
                figs4 = go.Figure(data = data,layout = layout)
                graphs4JSON = json.dumps(figs4, cls=plotly.utils.PlotlyJSONEncoder)


                # Histogram - Category
                # defining data
                trace = go.Histogram(
                    x=kctn['category'],
                    nbinsx=3,
                    marker = dict(color = '#1aafd0')
                    )
                data = [trace]
                # defining layout
                layout = go.Layout(title="Category Distribution")
                # defining figure and plotting
                figs5 = go.Figure(data = data,layout = layout)
                graphs5JSON = json.dumps(figs5, cls=plotly.utils.PlotlyJSONEncoder)


                #Churn
                lab = kctn["churn"].value_counts().keys().tolist()
                #values
                val = kctn["churn"].value_counts().values.tolist()
                trace = go.Pie(labels=lab, 
                                values=val, 
                                hole = 0.4,
                                # Seting values to 
                                hoverinfo="value")
                data = [trace]

                layout = go.Layout(dict(title="Churn",
                            plot_bgcolor = "white",
                            paper_bgcolor = "white",))
                figs6 = go.Figure(data = data,layout = layout)
                graphs6JSON = json.dumps(figs5, cls=plotly.utils.PlotlyJSONEncoder)


                # Churn Rate by Services
                plot_by_gender = kctn.groupby('services').churn.mean().reset_index()
                plot_data = [
                    go.Bar(
                        x=plot_by_gender['services'],
                        y=plot_by_gender['churn'],
                    width = [0.8],
                            marker = dict(
                            color=['#fc636b', '#ffb900', '#6a67ce', '#1aafd0', '#3be8b0']
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

                figs7 = go.Figure(data=plot_data, layout=layout)
                graphs7JSON = json.dumps(figs7, cls=plotly.utils.PlotlyJSONEncoder)


                # Churn Rate by Zone
                plot_by_payment = kctn.groupby('zone').churn.mean().reset_index()
                plot_data = [
                    go.Bar(
                        x=plot_by_payment['zone'],
                        y=plot_by_payment['churn'],
                    width = [0.8],
                        marker = dict(
                            color=['#fc636b', '#ffb900', '#6a67ce', '#1aafd0', '#3be8b0', '#fc636b', '#ffb900', '#6a67ce', '#1aafd0', '#3be8b0']
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

                figs8 = go.Figure(data=plot_data, layout=layout)
                graphs8JSON = json.dumps(figs8, cls=plotly.utils.PlotlyJSONEncoder)


                # Churn Rate by Category
                plot_by_contract = kctn.groupby('category').churn.mean().reset_index()
                plot_data = [
                    go.Bar(
                        x=plot_by_contract['category'],
                        y=plot_by_contract['churn'],
                        width = [0.8],
                        marker = dict(
                            color=['#fc636b', '#ffb900', '#6a67ce', '#1aafd0', '#3be8b0']
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

                figs9 = go.Figure(data=plot_data, layout=layout)
                graphs9JSON = json.dumps(figs9, cls=plotly.utils.PlotlyJSONEncoder)

                image_file = url_for('static', filename='images/' + current_user.image_file)
                return render_template("home.html", user= current_user, image_file=image_file,
                    
                    #  KCTN
                    graphs1JSON=graphs1JSON,
                    graphs2JSON=graphs2JSON,
                    graphs3JSON=graphs3JSON,
                    graphs4JSON=graphs4JSON,
                    graphs5JSON=graphs5JSON,
                    graphs6JSON=graphs6JSON,
                    graphs7JSON=graphs7JSON,
                    graphs8JSON=graphs8JSON,
                    graphs9JSON=graphs9JSON, active=active, disconnected=disconnected, total=total
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
            total = db.session.query(Otherdata).count()
            active = Otherdata \
                .query \
                .filter(Otherdata.status == "Active").count()
            disconnected = Otherdata \
                .query \
                .filter(Otherdata.status == "Disconnected").count()
            
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
                graph8JSON=graph8JSON,
                active=active,
                disconnected=disconnected, total=total,
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
    elif current_user.explore == "empty":

        image_file = url_for('static', filename='images/' + current_user.image_file)
        return render_template("home.html", user= current_user, image_file=image_file)
    image_file = url_for('static', filename='images/' + current_user.image_file)
    return render_template("home.html", user= current_user, image_file=image_file)

@views.route('/churn-analysis', methods=["GET", "POST"])
@login_required
def churnanalytics():
    if current_user.explore == "customer":
        if current_user.cname == "Kalibo Cable":
            kctn = pd.read_sql_table('data', con=cnx)
            kctn.head()

            # Convert dates to datetype
            kctn.activation_date = pd.to_datetime(kctn.activation_date)
            kctn.disconnection_date = pd.to_datetime(kctn.disconnection_date)
            kctn.reactivation_date = pd.to_datetime(kctn.reactivation_date)
            kctn.date_paid = pd.to_datetime(kctn.date_paid)

            kctn['disconnection_date'] = kctn['disconnection_date'].dt.strftime('%m-%d-%Y')
            kctn['reactivation_date'] = kctn['reactivation_date'].dt.strftime('%m-%d-%Y')
            kctn['activation_date'] = kctn['activation_date'].dt.strftime('%m-%d-%Y')
            kctn['date_paid'] = kctn['date_paid'].dt.strftime('%m-%d-%Y')

            # Remove Account Number-Address
            kctn2 = kctn.iloc[:,3:]
            # Remove Reference Number
            del kctn2['ref_no']
            del kctn2['date_paid']
            del kctn2['activation_date']
            del kctn2['disconnection_date']
            del kctn2['reactivation_date']

            # independent variable - all columns aside from 'Churn'
            X = kctn2.iloc[:,:-1].values
            # dependent variable - Churn
            y = kctn2.iloc[:,7]

            # Convert predictor variables in a binary numeric variable
            kctn2['status'].replace(to_replace='Active', value=1, inplace=True)
            kctn2['status'].replace(to_replace='Disconnected', value=0, inplace=True)

            # Converting categorical variables into dummy variables
            kctn_dummies = pd.get_dummies(kctn2)
            kctn_dummies.head()

            from sklearn.preprocessing import StandardScaler
            standardscaler = StandardScaler()
            columns_for_fit_scaling = ['monthly', 'amount_paid']
            kctn_dummies[columns_for_fit_scaling] = standardscaler.fit_transform(kctn_dummies[columns_for_fit_scaling])

            # Splitting Data into Train and Test
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,  random_state=101)

            y = kctn_dummies['churn'].values
            X = kctn_dummies.drop(columns = ['churn'])

            # Scaling all the variables to a range of 0 to 1
            from sklearn.preprocessing import MinMaxScaler
            features = X.columns.values
            scaler = MinMaxScaler(feature_range = (0,1))
            scaler.fit(X)
            X = pd.DataFrame(scaler.transform(X))
            X.columns = features

            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

            # Running logistic regression model
            from sklearn.linear_model import LogisticRegression
            import sklearn.metrics as metrics
            logmodel = LogisticRegression(random_state=50)
            result = logmodel.fit(X_train, y_train)

            Xnew = X_test.values

            proba = logmodel.predict_proba(Xnew)[:,1]
            for i in range(len(Xnew)):
                kctn['Churn Probability'] = proba[i]
                
            for i in range(len(Xnew)):
                kctn['Churn Probability'][i] = proba[i]
                predd = kctn[['account_no', 'amount_paid', 'monthly','Churn Probability']].values.tolist()
            cust = X_test.count()
            image_file = url_for('static', filename='images/' + current_user.image_file)
            return render_template("churn-analysis.html", user= current_user, image_file=image_file, my_list=predd, cust=cust)
        else:
            df = pd.read_sql_table('otherdata', con=cnx)
            dataf = df.loc[df['odata_id'] == current_user.id]
            row_count = dataf.index
            rc = len(row_count)
            if rc >= 3:
                # Check for missing values
                dataf.isna().any()

                # Convert dates to datetype
                dataf.activation_date = pd.to_datetime(dataf.activation_date)
                dataf.disconnection_date = pd.to_datetime(dataf.disconnection_date)
                dataf.reactivation_date = pd.to_datetime(dataf.reactivation_date)
                dataf.date_paid = pd.to_datetime(dataf.date_paid)

                dataf['disconnection_date'] = dataf['disconnection_date'].dt.strftime('%m-%d-%Y')
                dataf['reactivation_date'] = dataf['reactivation_date'].dt.strftime('%m-%d-%Y')
                dataf['activation_date'] = dataf['activation_date'].dt.strftime('%m-%d-%Y')
                dataf['date_paid'] = dataf['date_paid'].dt.strftime('%m-%d-%Y')

                # Calculate average and fill missing values
                na_cols = dataf.isna().any()
                na_cols = na_cols[na_cols == True].reset_index()
                na_cols = na_cols["index"].tolist()

                for col in dataf.columns[1:]:
                    if col in na_cols:
                        if dataf[col].dtype != 'object':
                            dataf[col] = dataf[col].fillna(dataf[col].mean()).round(0)


                # Label Encoder
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()

                # Label encoding for columns with 2 or less unique

                le_count = 0
                for col in dataf.columns[1:]:
                    if dataf[col].dtype == 'object':
                        if len(list(dataf[col].unique())) <=2:
                            le.fit(dataf[col])
                            dataf[col] = le.transform(dataf[col])
                            le_count +=1
                print('{} columns label encoded'.format(le_count))


                # Remove Account Number-Address
                dataf2 = dataf.iloc[:,3:]
                # Remove Reference Number
                del dataf2['ref_no']
                del dataf2['date_paid']
                del dataf2['activation_date']
                del dataf2['disconnection_date']
                del dataf2['reactivation_date']

                # independent variable - all columns aside from 'Churn'
                X = dataf.iloc[:,:-1].values
                X

                # dependent variable - Churn
                y = dataf.iloc[:,17]
                y

                # Remove customer ID
                dataf2 = dataf.iloc[:,1:]

                # Convert predictor variables in a binary numeric variable
                dataf2['churn'].replace(to_replace='Yes', value=1, inplace=True)
                dataf2['churn'].replace(to_replace='No', value=0, inplace=True)

                # Converting categorical variables into dummy variables
                dataf_dummies = pd.get_dummies(dataf2)
                dataf_dummies.head()

                from sklearn.preprocessing import StandardScaler
                standardscaler = StandardScaler()
                columns_for_fit_scaling = ['monthly', 'amount_paid']
                dataf_dummies[columns_for_fit_scaling] = standardscaler.fit_transform(dataf_dummies[columns_for_fit_scaling])

                # Splitting Data into Train and Test
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

                y = dataf_dummies['churn'].values
                X = dataf_dummies.drop(columns = ['churn'])

                # Scaling all the variables to a range of 0 to 1
                from sklearn.preprocessing import MinMaxScaler
                features = X.columns.values
                scaler = MinMaxScaler(feature_range = (0,1))
                scaler.fit(X)
                X = pd.DataFrame(scaler.transform(X))
                X.columns = features

                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

                # Running logistic regression model
                from sklearn.linear_model import LogisticRegression
                import sklearn.metrics as metrics
                logmodel = LogisticRegression(random_state=50)
                result = logmodel.fit(X_train, y_train)

                Xnew = X_test.values
                pred = logmodel.predict(X_test)

                proba = logmodel.predict_proba(Xnew)[:,1]
                logmodel_accuracy = round(metrics.accuracy_score(y_test, pred)*100, 2)
                print (logmodel_accuracy)

                for i in range(len(Xnew)):
                    dataf['Churn Probability'] = proba[i]

                for i in range(len(Xnew)):
                    dataf['Churn Probability'][i] = proba[i]
                predd = dataf[['account_no', 'amount_paid', 'monthly','Churn Probability']].values.tolist()
                cust = dataf['Churn Probability'].count()
                image_file = url_for('static', filename='images/' + current_user.image_file)
                return render_template("churn-analysis.html", user= current_user, image_file=image_file, my_list=predd, cust=cust)
            elif rc < 3 and rc >= 1:
                flash("Records must contain atleast 3 rows.", category="error")

                image_file = url_for('static', filename='images/' + current_user.image_file)
                return render_template("churn-analysis.html", user= current_user, image_file=image_file)
            elif rc < 1:
                flash("Add records in Customer Management.", category="error")
                image_file = url_for('static', filename='images/' + current_user.image_file)
                return render_template("churn-analysis.html", user= current_user, image_file=image_file)
    elif current_user.explore == "sample":
        df = pd.read_sql_table('sampledata', con=cnx)

        # Calculate average and fill missing values
        na_cols = df.isna().any()
        na_cols = na_cols[na_cols == True].reset_index()
        na_cols = na_cols["index"].tolist()

        for col in df.columns[1:]:
            if col in na_cols:
                if df[col].dtype != 'object':
                    df[col] = df[col].fillna(df[col].mean()).round(0)

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
        print('{} columns label encoded'.format(le_count))

        # independent variable - all columns aside from 'Churn'
        X = df.iloc[:,:-1].values
        X

        # dependent variable - Churn
        y = df.iloc[:,20]
        y

        # Remove customer ID
        df2 = df.iloc[:,1:]

        # Convert predictor variables in a binary numeric variable
        df2['Churn'].replace(to_replace='Yes', value=1, inplace=True)
        df2['Churn'].replace(to_replace='No', value=0, inplace=True)

        # Converting categorical variables into dummy variables
        df_dummies = pd.get_dummies(df2)
        df_dummies.head()

        #Perform One Hot Encoding using get_dummies method
        df= pd.get_dummies(df, columns = ['Contract','Dependents','DeviceProtection','gender',
                                                                'InternetService','MultipleLines','OnlineBackup',
                                                                'OnlineSecurity','PaperlessBilling','Partner',
                                                                'PaymentMethod','PhoneService','SeniorCitizen',
                                                                'StreamingMovies','StreamingTV','TechSupport'],
                                    drop_first=True)
        
        from sklearn.preprocessing import StandardScaler
        standardscaler = StandardScaler()
        columns_for_fit_scaling = ['tenure', 'MonthlyCharges', 'TotalCharges']
        df_dummies[columns_for_fit_scaling] = standardscaler.fit_transform(df_dummies[columns_for_fit_scaling])

        # Splitting Data into Train and Test
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

        y = df_dummies['Churn'].values
        X = df_dummies.drop(columns = ['Churn'])

        # Scaling all the variables to a range of 0 to 1
        from sklearn.preprocessing import MinMaxScaler
        features = X.columns.values
        scaler = MinMaxScaler(feature_range = (0,1))
        scaler.fit(X)
        X = pd.DataFrame(scaler.transform(X))
        X.columns = features

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

        # Running logistic regression model
        from sklearn.linear_model import LogisticRegression
        import sklearn.metrics as metrics
        logmodel = LogisticRegression(random_state=50)
        result = logmodel.fit(X_train, y_train)

        Xnew = X_test.values
        pred = logmodel.predict(X_test)
        proba = logmodel.predict_proba(Xnew)[:,1]
        logmodel_accuracy = round(metrics.accuracy_score(y_test, pred)*100, 2)
        print (logmodel_accuracy)

        for i in range(len(Xnew)):
            df['Churn Probability'] = proba[i]

        for i in range(len(Xnew)):
            df['Churn Probability'][i] = proba[i]
            if i <= len(Xnew):
                predd = df[['customerID' ,'Churn Probability']].values.tolist()
        cust = df['Churn Probability'].count()
        image_file = url_for('static', filename='images/' + current_user.image_file)
        return render_template("churn-analysis.html", user= current_user, image_file=image_file, my_list=predd, cust=cust)
    image_file = url_for('static', filename='images/' + current_user.image_file)
    return render_template("churn-analysis.html", user= current_user, image_file=image_file)
    
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

