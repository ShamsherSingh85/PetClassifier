from fastai.vision.all import *
import gradio as gr

learn = load_learner('export.pkl')

labels = [word.title() for word in [' '.join(word.split('_')) for word in learn.dls.vocab]]

def predict(img):
    img = PILImage.create(img)
    pred, pred_idx, probs = learn.predict(img)
    return {labels[i] : float(probs[i]) for i in range(len(labels))}

title = 'Pet Breed Classifier'
description = "A Pet Breed Classifier for most commonly found household pets using Oxford Pets Dataset"
article="<p style='text-align: center'><a href='https://tmabraham.github.io/blog/gradio_hf_spaces_tutorial' target='_blank'>Blog post</a></p>"
examples = ['pet_0.jpg','pet_1.jpg','pet_2.jpg','pet_3.jpg','pet_4.jpg']
interpretation = 'default'
enable_queue = True 

gr.Interface(fn = predict, inputs = gr.Image(), 
             outputs = gr.Label(num_top_classes = 3),
             title = title, 
             description = description, 
             article = article, 
             examples = examples).queue().launch(share = True)