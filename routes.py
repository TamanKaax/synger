from app import app, germix_fastai_webapp
import os
import glob
from flask import request, redirect, url_for, render_template, session, flash
from app.forms import SelectForm, ConfirmForm, ResultForm
from werkzeug.utils import secure_filename
import numpy as np
import matplotlib.pyplot as plt
import urllib
from io import BytesIO
import base64
import numpy as np
from PIL import Image, ImageDraw, ImageFont

app.config['SECRET_KEY'] = 'dit is None of you r Bus'

def get_frame_overview(ch, setups=None):
    if not setups:
        setups = germix_fastai_webapp.logic_setups_only(f'{app.static_folder}/{ch}')
    imgs = sorted([f for f in os.listdir(f'{app.static_folder}/{ch}') if f[-4:] == '.jpg'])
    img = f'{app.static_folder}/{ch}/{imgs[0]}'
    img_io = BytesIO()
    imgpass = Image.open(img)
    f = 1
    for s in setups:
        (y, x) = (round(s[0]+(s[1]-s[0])/2), round(s[2]+(s[3]-s[2])/2))
        draw = ImageDraw.Draw(imgpass)
        font = ImageFont.truetype(f'{app.static_folder}/Roboto-Bold.ttf', size=250)
        draw.text((x, y), str(f), fill= 'rgb(255, 255, 255)', font=font)
        f += 1
    imgpass.save(img_io, format='PNG')
    img_io.seek(0)
    return urllib.parse.quote(base64.b64encode(img_io.read()).decode()), len(imgs)

@app.route('/', methods=['GET', 'POST'])
@app.route('/select')
def select():
    text = ['Loading an Image stack may take a few seconds']
    form = SelectForm()
    if form.validate_on_submit():
        session['ch'] = form.choice.data
        return redirect(url_for('confirm'))
    return render_template('select.html', title='Germix Selection Menu', form=form, text=text)

@app.route('/confirm', methods=['GET', 'POST'])
def confirm():
    form = ConfirmForm()
    ch = session['ch']
    #If I don't put following code in if-statement, it runs again when redirecting, why?
    if ch:
        frame_overview, len_imgs = get_frame_overview(ch)
        text = [f'ImageStack {ch} selected', f'Processing may take up to {int(np.round(len_imgs)*2/60)} minutes per frame']
    if form.validate_on_submit():
        if form.change.data:
            return redirect(url_for('select'))
        if form.proc.data:
            session['frame'] = form.frame.data
            return redirect(url_for('results'))
    return render_template('confirm.html', title='Confirm Choice', frame_overview=frame_overview, form=form, text=text)

@app.route('/results', methods=['GET', 'POST'])
def results():
    frame = session.pop('frame', 'All')
    if frame == 'All':
        begin_frame = int(0)
        end_frame = int(6)
    else:
        end_frame = int(frame)
        begin_frame = int(frame) - 1
    ch = session.pop('ch', None)
    plot_data = []
    if ch:
        results, setups = germix_fastai_webapp.logic(f'{app.static_folder}/{ch}', interval=1, begin_frame=begin_frame, end_frame=end_frame, show=False)
        frame_overview, _ = get_frame_overview(ch, setups=setups)
        for j in range(begin_frame, end_frame):
            img = BytesIO()
            fig, ax = plt.subplots()
            ax.plot(list(results[j].keys()), list(results[j].values()))
            ax.set_xlabel('Time')
            ax.set_ylabel('Quantity')
            ax.set_title(f'Germination graph for frame {j+1}')
            plt.savefig(img, format='png')
            img.seek(0)
            plot_data.append(urllib.parse.quote(base64.b64encode(img.read()).decode()))
    form = ResultForm()
    text = [f'Results for image Stack {ch}']
    if form.validate_on_submit():
        if form.export.data:
            pass
        if form.other.data:
            return redirect(url_for('select'))
    return render_template('results.html', title='Results', plot_url=plot_data, form=form, text=text, frame_overview=frame_overview)
