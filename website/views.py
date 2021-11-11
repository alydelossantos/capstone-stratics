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


        churners_number = len(df[df['Churn'] == 1])

        churners = (df[df['Churn'] == 1])

        non_churners = df[df['Churn'] == 0].sample(n=churners_number)
        df2 = churners.append(non_churners)

        try:
            customer_id = df2['customerID'] # Store this as customer_id variable
            del df2['customerID'] # Don't need in ML DF
        except:
            print("already removed customerID")


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

        logmodel_accuracy = round(metrics.accuracy_score(y_test, pred)*100, 2)

        proba = logmodel.predict_proba(Xnew)[:,1]

        for i in range(len(Xnew)):
	        df['Churn Probability'][i] = proba[i]

        display = df[(df['Churn Probability'].isnull())].index
        df.drop(display, inplace=True)

        # Create a Dataframe showcasing probability of Churn of each customer
        df[['customerID','Churn Probability']]

        # ---------- Decision Tree -------------

        from sklearn.tree import DecisionTreeClassifier
        dtmodel = DecisionTreeClassifier(criterion = 'gini', random_state=50)
        dtmodel.fit(X_train, y_train)

        dt_pred = dtmodel.predict(X_test)

        dt_accuracy = round(metrics.accuracy_score(y_test, dt_pred)*100,2)


        # ---------- RandomForest ----------------

        from sklearn.ensemble import RandomForestClassifier
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
        rfmodel = RandomForestClassifier(n_estimators=1000 , oob_score = True, n_jobs = -1,
                                        random_state =50, max_features = "auto",
                                        max_leaf_nodes = 30)
        rfmodel.fit(X_train, y_train)

        # Make predictions
        rf_pred = rfmodel.predict(X_test)

        rf_accuracy = round(metrics.accuracy_score(y_test, rf_pred)*100,2)



        # ------------- XGBoost -------------------

        from xgboost import XGBClassifier
        model = XGBClassifier()
        model.fit(X_train, y_train)
        xgb_pred = model.predict(X_test)
        metrics.accuracy_score(y_test, xgb_pred)

        xgb_accuracy = round(metrics.accuracy_score(y_test, xgb_pred)*100,2)

        # ----------- Model Comparison --------------

        Model_Comparison = pd.DataFrame({
            'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost'],
            'Score': [logmodel_accuracy, dt_accuracy, rf_accuracy, xgb_accuracy]})
        Model_Comparison_df = Model_Comparison.sort_values(by='Score', ascending=False)
        Model_Comparison_df - Model_Comparison_df.set_index('Score')
        Model_Comparison_df.reset_index()
	
        logtrainpred = logmodel.predict(X_train)
        logtrainpred

        algomodels = [logmodel_accuracy, dt_accuracy, rf_accuracy,  xgb_accuracy]

        max_algo = np.max(algomodels)

        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import classification_report, accuracy_score

        algomodels = [logmodel_accuracy, dt_accuracy, rf_accuracy,  xgb_accuracy]

        max_algo = np.max(algomodels)

        if max_algo == logmodel_accuracy:
            y_hat_train = logmodel.predict(X_train)
            y_hat_test = logmodel.predict(X_test)
            
            conf_mat_logmodel = confusion_matrix(y_test,y_hat_test)
            conf_mat_logmodel

        elif max_algo == dt_accuracy:
            y_hat_train = dtmodel.predict(X_train)
            y_hat_test = dtmodel.predict(X_test)
        
            conf_mat_dtmodel = confusion_matrix(y_test,y_hat_test)
            conf_mat_dtmodel

        elif max_algo == rf_accuracy:
            y_hat_train = rfmodel.predict(X_train)
            y_hat_test = rfmodel.predict(X_test)

            conf_mat_rfmodel = confusion_matrix(y_test,y_hat_test)
            conf_mat_rfmodel 
            
        elif max_algo == xgb_accuracy:
            y_hat_train = model.predict(X_train)
            y_hat_test = model.predict(X_test)

            conf_mat_xgbmodel = confusion_matrix(y_test,y_hat_test)
            conf_mat_xgbmodel    
            
        return render_template('model.html', tables=[Model_Comparison.to_html(classes='data')], titles=Model_Comparison.columns.values)
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

    elif current_user.explore == "customer":
        if current_user.cname.lower() == kfull.lower() or current_user.cname.lower() == knoinc.lower() or current_user.cname.lower() == knonet.lower() or current_user.cname.lower() == knotel.lower() or current_user.cname.lower() == knocable.lower() or current_user.cname.lower() == abbrenoinc.lower():
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
                    graph24JSON=graph24JSON)
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
                # independent variable
                X = dataf.iloc[:,:1].values
                X

                # dependent variable - churn column
                y = dataf.iloc[:,8]
                y

                # Counts number of null values - resulted that no values are missing.
                null_columns=df.columns[dataf.isnull().any()]
                dataf[null_columns].isnull().sum()

                # Splitting Data into Train and Test
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

                # Zscore
                from scipy import stats
                zscore = np.abs(stats.zscore(df['monthly']))

                # zscore values higher than 3 are outliers.
                threshold = 3

                dataf.corr(method='pearson')

                # Create Pivot Table - compute for sum
                pd.pivot_table(dataf, index=['address', 'services'], aggfunc = 'sum')

                # Create Pivot Table - compute for mean
                pd.pivot_table(dataf, index=['address', 'services'], aggfunc = 'mean')    

                # Create Pivot Table - compute for count
                pd.pivot_table(dataf, index=['address', 'services'], aggfunc = 'count')

                # Pie Chart
                from plotly.offline import init_notebook_mode,iplot
                import plotly.graph_objects as go
                import cufflinks as cf
                init_notebook_mode(connected=True)

                #labels
                lab = dataf["collector"].value_counts().keys().tolist()
                #values
                val = dataf["collector"].value_counts().values.tolist()
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
                trace = go.Histogram(x=dataf['services'],nbinsx=40,histnorm='percent')
                data = [trace]
                # defining layout
                layout = go.Layout(title="Service Distribution")
                # defining figure and plotting
                fig2 = go.Figure(data = data,layout = layout)
                graph2JSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

                # Histogram - State
                # defining data
                trace = go.Histogram(x=dataf['address'],nbinsx=52)
                data = [trace]
                # defining layout
                layout = go.Layout(title="Address")
                # defining figure and plotting
                fig3 = go.Figure(data = data,layout = layout)
                fig3 = go.Figure(data = data,layout = layout)
                graph3JSON = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)

                # Histogram - Churn
                # defining data
                trace = go.Histogram(x=dataf['sstatus'],nbinsx=3)
                data = [trace]
                # defining layout
                layout = go.Layout(title="Churn Distribution")
                # defining figure and plotting
                fig4 = go.Figure(data = data,layout = layout)
                fig4 = go.Figure(data = data,layout = layout)
                graph4JSON = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)

                current_user.dash = "full"
                db.session.add(current_user)
                db.session.commit()
                image_file = url_for('static', filename='images/' + current_user.image_file)
                return render_template("home.html", user= current_user, image_file=image_file, graph1JSON=graph1JSON, 
                graph2JSON=graph2JSON, 
                graph3JSON=graph3JSON,
                graph4JSON=graph4JSON)
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

