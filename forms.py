from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField
from wtforms.validators import DataRequired
class InputForm(FlaskForm):
    input_sent = TextAreaField('input_amr', validators=[DataRequired()])
