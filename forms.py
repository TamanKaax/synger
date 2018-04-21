from flask_wtf import FlaskForm
from wtforms import SelectField, SubmitField

class SelectForm(FlaskForm):
    choice = SelectField('Choose ImageStack', choices=[('tom1', 'tom1'), ('tom2', 'tom2'), ('tom3', 'tom3')])
    #process or preview are keywords and lead to errors
    proc = SubmitField('Process')
    prev = SubmitField('Preview')

class ConfirmForm(FlaskForm):
    proc = SubmitField('Process')
    change = SubmitField('Change')

class ResultForm(FlaskForm):
    export = SubmitField('Export')
    other = SubmitField('Other')
