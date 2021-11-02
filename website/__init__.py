from flask import Flask

from .commands import create_tables
from flask_sqlalchemy import SQLAlchemy
from os import path
from flask_login import LoginManager
from flask_mail import Mail

db = SQLAlchemy()
mail = Mail()
DB_NAME = "db.db"

def create_app(config_file='configure.py'): #create database
    app = Flask(__name__)
    app.config.from_pyfile(config_file)
    
    db.init_app(app)
    
    mail.init_app(app)
    
    from .views import views
    from .auth import auth

    app.register_blueprint(views, url_prefix = '/')
    app.register_blueprint(auth, url_prefix = '/')

    from .models import User, Data, Strategies, Contact, Sampledata, Samplestrategies

    login_manager = LoginManager() #user verification
    login_manager.login_view = "auth.signin"
    login_manager.init_app(app)
    
    @login_manager.user_loader
    def load_user(id):
        return User.query.get(int(id))
    
    app.cli.add_command(create_tables)
        
    return app   
