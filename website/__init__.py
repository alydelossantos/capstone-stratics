from flask import Flask

from .command import create_tables
from .extensions import db
from .models import User, Data, Otherdata, Sampledata, Strategies, Otherstrategies, Samplestrategies, Contact, Task
from flask_login import LoginManager
from flask_mail import Mail

mail = Mail()

#CREATE DATABASE
def create_app(config_file='configure.py'):
    app = Flask(__name__)
    app.config.from_pyfile(config_file)
    
    db.init_app(app)
    
    #SEND EMAILS
    mail.init_app(app)
    
   
    from .models import User, Data, Otherdata, Sampledata, Strategies, Otherstrategies, Samplestrategies, Contact, Task
    
    from .views import views
    from .auth import auth

    app.register_blueprint(views, url_prefix = '/')
    app.register_blueprint(auth, url_prefix = '/')
    
    #LOGIN AUTHENTICATION
    login_manager = LoginManager() #user verification
    login_manager.login_view = "auth.signin"
    login_manager.init_app(app)
    
    @login_manager.user_loader
    def load_user(id):
        return User.query.get(int(id))
    
    #CREATE TABLES
    app.cli.add_command(create_tables)
        
    return app   
