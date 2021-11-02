import click
from flask import Flask
from flask.cli import with_appcontext

from flask_sqlalchemy import SQLAlchemy
app = Flask(__name__)
db = SQLAlchemy(app)
from .models import User, Data, Strategies, Contact, Sampledata, Samplestrategies

@click.command(name='create_tables')
@with_appcontext
def create_tables():
  db.create_all()
