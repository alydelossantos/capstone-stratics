import os
import re

MAIL_SERVER = 'smtp.gmail.com'
MAIL_PORT = 587
MAIL_USE_TLS = True
MAIL_USE_SSL = False
MAIL_USERNAME = 'horizonfeua@gmail.com'
MAIL_PASSWORD = 'sleepdeprived'
MAIL_DEFAULT_SENDER = 'horizonfeua@gmail.com'

uri = os.getenv('DATABASE_URL')
if uri and uri.startswith('postgres://'):
  uri = uri.replace('postgres://', 'postgresql://', 1)
  
SECRET_KEY = 'asdfghjkl'
SQLALCHEMY_DATABASE_URI = uri
SQLALCHEMY_TRACK_MODIFICATIONS = False
