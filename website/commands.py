import click
from flask.cli import with_appcontext
from flask_sqlalchemy import SQLAlchemy

from .models import User, Data, Strategies, Contact, Sampledata, Samplestrategies

@click.command(name='create_tables')
@with_appcontext
def create_tables():
    db.create_all()