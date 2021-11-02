import os

MAIL_SERVER = 'smtp.gmail.com'
MAIL_PORT = 465
MAIL_USE_TLS = False
MAIL_USE_SSL = True
MAIL_USERNAME = 'horizonfeua@gmail.com'
MAIL_PASSWORD = 'horizonfeu123'
MAIL_DEFAULT_SENDER = 'horizonfeua@gmail.com'

SECRET_KEY = 'asdfghjkl'
SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL')
SQLALCHEMY_TRACK_MODIFICATIONS = False
