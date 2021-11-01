import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')

MAIL_SERVER = 'smtp.gmail.com'
MAIL_PORT = 465
MAIL_USE_TLS = False
MAIL_USE_SSL = True
MAIL_USERNAME = 'horizonfeua@gmail.com'
MAIL_PASSWORD = 'horizonfeu123'
MAIL_DEFAULT_SENDER = 'horizonfeua@gmail.com'