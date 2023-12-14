from flask import Flask, render_template, request
import torchvision.models as models
import torch
import subprocess

from utils import *

app = Flask(__name__)

def get_model_options():
    model_names = sorted(name for name in models.__dict__
                        if name.islower() and not name.startswith("__")
                        and callable(models.__dict__[name]))
    return model_names

@app.route('/')
def index():
    model_options = get_model_options()
    return render_template('index.html', model_options=model_options)

@app.route('/submit', methods=['POST'])
def submit():
    # Retrieve form data
    dataset_path = request.form['dataset_path']
    arch = request.form['model']

    pruning_epochs = int(request.form['pruning_epochs'])
    pruning_workers = int(request.form['pruning_workers'])
    pruning_batch_size = int(request.form['pruning_batch_size'])
    pruning_learning_rate = float(request.form['pruning_learning_rate'])
    pruning_momentum = float(request.form['pruning_momentum'])
    pruning_weight_decay = float(request.form['pruning_weight_decay'])
    pruning_sparsity = float(request.form['pruning_sparsity'])

    distillation_epochs = int(request.form['distillation_epochs'])
    distillation_workers = int(request.form['distillation_workers'])
    distillation_batch_size = int(request.form['distillation_batch_size'])
    distillation_learning_rate = float(request.form['distillation_learning_rate'])
    distillation_momentum = float(request.form['distillation_momentum'])
    distillation_weight_decay = float(request.form['distillation_weight_decay'])
    teacher_model = request.form['teacher_model']

    quantization_epochs = int(request.form['quantization_epochs'])
    quantization_workers = int(request.form['quantization_workers'])
    quantization_batch_size = int(request.form['quantization_batch_size'])
    quantization_learning_rate = float(request.form['quantization_learning_rate'])
    quantization_momentum = float(request.form['quantization_momentum'])
    quantization_weight_decay = float(request.form['quantization_weight_decay'])
    per_channel = request.form.get('per_channel', False)

    # Process the data or perform any other actions based on the form inputs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if per_channel == False:
        model = model_prepaper(arch, device, 1)

    else:
        model = model_prepaper(arch, device, 2)

    torch.save(model, arch + ".pth")
    
    subprocess.run(["python","prune.py", '--arch', arch,'--no-pretrained', 
                    '--model_path', arch + ".pth", '--epochs', str(pruning_epochs),
                    '--batch-size', str(pruning_batch_size), '--workers', str(pruning_workers),
                    '--lr', str(pruning_learning_rate), '--momentum', str(pruning_momentum),
                    '--wd', str(pruning_weight_decay), '--sparsity', str(pruning_sparsity), 
                    str(dataset_path)])

    subprocess.run(["python","distillation.py", '--sa', arch, '--ta', teacher_model, '--no-pretrained', 
                    '--model_path', arch + ".pth", '--epochs', str(distillation_epochs),
                    '--workers', str(distillation_workers), '--batch-size', str(distillation_batch_size),
                    '--lr', str(distillation_learning_rate), '--momentum', str(distillation_momentum),
                    '--wd', str(distillation_weight_decay),
                    str(dataset_path)])

    subprocess.run(["python","quantization.py", '--arch', arch,'--no-pretrained',
                    '--model_path', arch + ".pth", '--epochs', str(quantization_epochs),
                    '--workers', str(quantization_workers), '--batch-size', str(quantization_batch_size),
                    '--lr', str(quantization_learning_rate), '--momentum', str(quantization_momentum),
                    '--wd', str(quantization_weight_decay),
                    str(dataset_path)])



    return render_template('result.html', result=arch + ".ts")    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
