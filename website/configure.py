import os
import re

MAIL_SERVER = 'smtp.gmail.com'
MAIL_PORT = 465
MAIL_USE_TLS = False
MAIL_USE_SSL = True
MAIL_USERNAME = 'horizonfeua@gmail.com'
MAIL_PASSWORD = 'sleepdeprived'
MAIL_DEFAULT_SENDER = 'horizonfeua@gmail.com'

uri = os.getenv('DATABASE_URL')
if uri and uri.startswith('postgres://'):
  uri = uri.replace('postgres://', 'postgresql://', 1)
  
SECRET_KEY = 'asdfghjkl'
SQLALCHEMY_DATABASE_URI = uri
SQLALCHEMY_TRACK_MODIFICATIONS = False
