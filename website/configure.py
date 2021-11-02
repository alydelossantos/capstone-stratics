import os
import re

MAIL_SERVER = 'smtp.gmail.com'
MAIL_PORT = 465
MAIL_USE_TLS = False
MAIL_USE_SSL = True
MAIL_USERNAME = 'ksn.080900@gmail.com'
MAIL_PASSWORD = 's4noope@cH'
MAIL_DEFAULT_SENDER = 'ksn.080900@gmail.com'

uri = os.getenv('DATABASE_URL')
if uri and uri.startswith('postgres://'):
  uri = uri.replace('postgres://', 'postgresql://', 1)
  
SECRET_KEY = 'asdfghjkl'
SQLALCHEMY_DATABASE_URI = uri
SQLALCHEMY_TRACK_MODIFICATIONS = False
