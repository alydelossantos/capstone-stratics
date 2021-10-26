from flask import Blueprint, render_template
from flask_login import login_user, login_required, logout_user, current_user

views = Blueprint('views', __name__)

@views.route('/')
def landing():
    return render_template("landing.html", user= current_user)


@views.route('/home')
@login_required
def home():
    return render_template("home.html", user= current_user)
    

