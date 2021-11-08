from flask import Flask

from .command import create_tables
from .extensions import db
from .models import User, Data, Strategies, Contact, Sampledata, Otherdata, Otherstrategies, Task
from flask_login import LoginManager
from flask_mail import Mail

mail = Mail()

def create_app(config_file='configure.py'): #create database
    app = Flask(__name__)
    app.config.from_pyfile(config_file)
    
    db.init_app(app)
    
    mail.init_app(app)
    
   
    from .models import User, Data, Strategies, Contact, Sampledata, Otherdata, Otherstrategies, Task
    
    from .views import views
    from .auth import auth

    app.register_blueprint(views, url_prefix = '/')
    app.register_blueprint(auth, url_prefix = '/')

    login_manager = LoginManager() #user verification
    login_manager.login_view = "auth.signin"
    login_manager.init_app(app)
    
    @login_manager.user_loader
    def load_user(id):
        return User.query.get(int(id))
    
    app.cli.add_command(create_tables)
        
    return app   
