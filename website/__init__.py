from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from os import path
from flask_login import LoginManager
from flask_mail import Mail

db = SQLAlchemy()
mail = Mail()
DB_NAME = "db.db"

def create_app(): #create database
    app = Flask(__name__)
    app.config.from_pyfile('config.cfg')
    
    db.init_app(app)
    
    mail.init_app(app)
    
    from .views import views
    from .auth import auth

    app.register_blueprint(views, url_prefix = '/')
    app.register_blueprint(auth, url_prefix = '/')

    from .models import User, Data, Strategies, Contact, Sampledata, Samplestrategies

    create_database(app)
   
    login_manager = LoginManager() #user verification
    login_manager.login_view = "auth.signin"
    login_manager.init_app(app)
    
    @login_manager.user_loader
    def load_user(id):
        return User.query.get(int(id))
    
        
    return app


