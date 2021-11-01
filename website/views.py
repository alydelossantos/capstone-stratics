from flask import Blueprint, render_template, request, url_for
from flask_login import login_user, login_required, logout_user, current_user

views = Blueprint('views', __name__)

@views.route('/home', methods=["GET", "POST"])
@login_required
def home():
    if current_user.explore == "sample":
        current_user.dname = "Sample Dataset"
    else:
        current_user.dname = "Customer Dataset"
    print(current_user.dname)
    image_file = url_for('static', filename='images/' + current_user.image_file)
    return render_template("home.html", user= current_user)

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

    
