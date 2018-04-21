from flask_wtf import FlaskForm
from wtforms import SelectField, SubmitField

class SelectForm(FlaskForm):
    choice = SelectField('Choose ImageStack', choices=[('tom1', 'tom1'), ('tom2', 'tom2'), ('tom3', 'tom3')])
    #process or preview are keywords and lead to errors
    select = SubmitField('Select')

class ConfirmForm(FlaskForm):
    frame = SelectField('Choose frame(s)', choices = [('1', '1'), ('2', '2'), ('3', '3'), ('4', '4'), ('5', '5'), ('6', '6'), ('All', 'All')], 
            default = 'All')
    proc = SubmitField('Process')
    change = SubmitField('Back')

class ResultForm(FlaskForm):
    export = SubmitField('Export')
    other = SubmitField('Other')
