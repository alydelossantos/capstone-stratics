import click
from flask import Flask
from flask.cli import with_appcontext

from .extensions import db

from .models import User, Data, Strategies, Contact, Sampledata, Otherdata, Otherstrategies

@click.command(name='create_tables')
@with_appcontext
def create_tables():
  db.create_all()