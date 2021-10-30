from website import db
from flask_login import UserMixin
from sqlalchemy.sql import func
    
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    fname = db.Column(db.String(50), nullable=False , default='N/A')
    lname = db.Column(db.String(50), nullable=False , default='N/A')
    uname = db.Column(db.String(50), nullable=False , default='N/A')
    email = db.Column(db.String(100), unique=True, nullable=False)
    cname = db.Column(db.String(50), nullable=False , default='N/A')
    address = db.Column(db.String(50), nullable=False , default='N/A')
    cp = db.Column(db.String(50), nullable=False , default='N/A')
    fb = db.Column(db.String(50), nullable=False , default='N/A')
    ig = db.Column(db.String(50), nullable=False , default='N/A')
    tw = db.Column(db.String(50), nullable=False , default='N/A')
    linkedin = db.Column(db.String(50), nullable=False , default='N/A')
    bday = db.Column(db.String(50), nullable=False , default='N/A')
    about = db.Column(db.String(500), nullable=False , default='N/A')
    password = db.Column(db.String(50), nullable=False , default='N/A')
    image_file = db.Column(db.String(20), nullable=False, default='user.png')
    data = db.relationship("Data")
    strategies = db.relationship("Strategies")
    sample_strat = db.relationship("Samplestrategies")
    contacts = db.relationship("Contact")
    sample_data = db.relationship("Sampledata")
    
        
class Data(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    duser_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    accnt_num = db.Column(db.String(100))
    name = db.Column(db.String(100))
    address = db.Column(db.String(100))
    services = db.Column(db.String(100))
    monthly = db.Column(db.String(100))
    collector = db.Column(db.String(100))
    sstatus = db.Column(db.String(150))
    amnt_paid = db.Column(db.String(150))
    ref_num = db.Column(db.String(100))

class Sampledata(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    sduser_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    accnt_num = db.Column(db.String(100))
    name = db.Column(db.String(100))
    address = db.Column(db.String(100))
    services = db.Column(db.String(100))
    monthly = db.Column(db.String(100))
    collector = db.Column(db.String(100))
    sstatus = db.Column(db.String(150))
    amnt_paid = db.Column(db.String(150))
    ref_num = db.Column(db.String(100))
    
class Strategies(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    stratuser_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    name = db.Column(db.String(100))
    act = db.Column(db.String(100))
    platform = db.Column(db.String(100))
    startdate = db.Column(db.String(150))
    enddate = db.Column(db.String(150))
    status = db.Column(db.String(100))
    description = db.Column(db.Text(225))
    
class Samplestrategies(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    sstratuser_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    name = db.Column(db.String(100))
    act = db.Column(db.String(100))
    platform = db.Column(db.String(100))
    startdate = db.Column(db.String(150))
    enddate = db.Column(db.String(150))
    status = db.Column(db.String(100))
    description = db.Column(db.Text(225))
 
class Contact(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    cuser_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    name = db.Column(db.String(100))
    email = db.Column(db.String(100))
    message = db.Column(db.Text(225))
