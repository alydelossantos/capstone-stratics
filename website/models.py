from .extensions import db
from flask_login import UserMixin
from sqlalchemy.sql import func
from website.configure import SECRET_KEY
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    fname = db.Column(db.String(50))
    lname = db.Column(db.String(50))
    uname = db.Column(db.String(50))
    email = db.Column(db.String(100), unique=True)
    cname = db.Column(db.String(50), default='Company')
    address = db.Column(db.String(50), default='Philippines')
    cp = db.Column(db.String(50), default='09*********')
    fb = db.Column(db.String(50), default='Facebook')
    ig = db.Column(db.String(50), default='Instagram')
    tw = db.Column(db.String(50), default='Twitter')
    linkedin = db.Column(db.String(50), default='LinkedIn')
    bday = db.Column(db.String(50), default='mm/dd/yyyy')
    about = db.Column(db.String(500), default='Describe Yourself')
    password = db.Column(db.String(50))
    image_file = db.Column(db.String(20), default='default.png') 
    dname = db.Column(db.String(100), nullable=False, default='Dashboard Name')
    explore = db.Column(db.String(50), nullable=False, default='empty')
    email_confirmation_sent_on = db.Column(db.DateTime, nullable=True)
    email_confirmed = db.Column(db.Boolean, nullable=True, default=False)
    email_confirmed_on = db.Column(db.DateTime, nullable=True)
    user_type = db.Column(db.String(50))
    user_status = db.Column(db.Boolean, nullable=True, default=False)
    position = db.Column(db.String(50), default="Position")
    dash = db.Column(db.String(20), default="none")
    ccode = db.Column(db.String(20))
    other_data = db.relationship("Otherdata")
    other_strategies = db.relationship("Otherstrategies")
    
    def get_reset_token(self, expires_sec=1800):
        s = Serializer(SECRET_KEY, expires_sec)
        return s.dumps({'user_id':self.id}).decode('utf-8')
    
    @staticmethod
    def verify_reset_token(token):
        s = Serializer(SECRET_KEY)
        try:
            user_id = s.loads(token)['user_id']
        except:
            return None
        return User.query.get(user_id)
    
class Data(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    account_no = db.Column(db.String(100))
    subscriber = db.Column(db.String(100))
    address = db.Column(db.String(225))
    zone = db.Column(db.String(50))
    services = db.Column(db.String(100))
    monthly = db.Column(db.Numeric)
    collector = db.Column(db.String(50))
    status = db.Column(db.String(50))
    amount_paid = db.Column(db.Numeric)
    ref_no = db.Column(db.String(100), nullable=True)
    date_paid = db.Column(db.String(50), nullable=True)
    category = db.Column(db.String(50))
    activation_date = db.Column(db.String(100))
    disconnection_date = db.Column(db.String(100), nullable=True)
    reactivation_date = db.Column(db.String(100), nullable=True)
    churn = db.Column(db.Numeric)
    
    def to_dict(self):
        return {
            'account_no': self.account_no,
            'subscriber': self.subscriber,
            'address' : self.address
        }
    
class Otherdata(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    odata_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    account_no = db.Column(db.String(100))
    subscriber = db.Column(db.String(100))
    gender = db.Column(db.String(20))
    address = db.Column(db.String(225))
    province = db.Column(db.String(100))
    services = db.Column(db.String(100))
    monthly = db.Column(db.Numeric)
    status = db.Column(db.String(50))
    amount_paid = db.Column(db.Numeric)
    ref_no = db.Column(db.String(100), nullable=True)
    date_paid = db.Column(db.String(50), nullable=True)
    category = db.Column(db.String(50))
    activation_date = db.Column(db.String(100))
    disconnection_date = db.Column(db.String(100), nullable=True)
    reactivation_date = db.Column(db.String(100), nullable=True)
    churn = db.Column(db.Numeric)
    
class Sampledata(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    customerID = db.Column(db.String(100))
    gender = db.Column(db.String(20))
    SeniorCitizen = db.Column(db.String(20))
    Partner = db.Column(db.String(20))
    Dependents = db.Column(db.String(20))
    tenure = db.Column(db.Integer)
    PhoneService = db.Column(db.String(20))
    MultipleLines = db.Column(db.String(20))
    InternetService = db.Column(db.String(100))
    OnlineSecurity = db.Column(db.String(20))
    OnlineBackup = db.Column(db.String(20))
    DeviceProtection = db.Column(db.String(20))
    TechSupport = db.Column(db.String(20))
    StreamingTV = db.Column(db.String(20))
    StreamingMovies = db.Column(db.String(20))
    Contract = db.Column(db.String(20))
    PaperlessBilling = db.Column(db.String(20))
    PaymentMethod = db.Column(db.String(50))
    MonthlyCharges = db.Column(db.Numeric)
    TotalCharges = db.Column(db.Numeric)
    Churn = db.Column(db.String(20))
  
class Strategies(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    name = db.Column(db.String(100))
    act = db.Column(db.String(100))
    platform = db.Column(db.String(100))
    startdate = db.Column(db.String(150))
    enddate = db.Column(db.String(150))
    status = db.Column(db.String(100))
    description = db.Column(db.String(225))
    
class Otherstrategies(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    ostrat_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    name = db.Column(db.String(100))
    act = db.Column(db.String(100))
    platform = db.Column(db.String(100))
    startdate = db.Column(db.String(150))
    enddate = db.Column(db.String(150))
    status = db.Column(db.String(100))
    description = db.Column(db.String(225))
    
class Samplestrategies(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    name = db.Column(db.String(100))
    act = db.Column(db.String(100))
    platform = db.Column(db.String(100))
    startdate = db.Column(db.String(150))
    enddate = db.Column(db.String(150))
    status = db.Column(db.String(100))
    description = db.Column(db.String(225))
    
class Contact(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(100), unique=True)
    message = db.Column(db.String(225))
    
class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text)

    def __init__(self, content):
        self.content = content
        self.done = False

    def __repr__(self):
        return '<Content %s>' % self.content

