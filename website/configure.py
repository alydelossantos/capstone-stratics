import os

MAIL_SERVER = 'smtp.gmail.com'
MAIL_PORT = 465
MAIL_USE_TLS = False
MAIL_USE_SSL = True
MAIL_USERNAME = 'horizonfeua@gmail.com'
MAIL_PASSWORD = 'horizonfeu123'
MAIL_DEFAULT_SENDER = 'horizonfeua@gmail.com'

SECRET_KEY = 'asdfghjkl'
SQLALCHEMY_DATABASE_URI = os.environ.get('postgres://jzyiaknneqredi:b3f16c49a8b520b2d627ba916908f41bc0a507f7cac2efcb23fa3a8947d76fa8@ec2-35-169-43-5.compute-1.amazonaws.com:5432/dc0chgkng9ougq')
SQLALCHEMY_TRACK_MODIFICATIONS = False
