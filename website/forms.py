from re import I
from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed, FileField
from flask_login import current_user
from flask_wtf.recaptcha import validators
from wtforms import StringField, PasswordField, SubmitField, BooleanField, TextField, TextAreaField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError
from website.models import User


class RequestResetForm(FlaskForm):
    email = StringField ('Email', validators = [DataRequired(), Email()])
    submit = SubmitField('Request Password Reset')

    
    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user is None:
            raise ValidationError('Email doest not exist')

class ResetPasswordForm(FlaskForm):
    password = PasswordField('Password',validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(),EqualTo('password')])
    submit = SubmitField('Reset Password')
