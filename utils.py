import datetime
import time
import sys
import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score

EPOCHS = 10
LEARNING_RATE = 2e-5
EPS = 1e-8
WARMUP = 100

def update_progress(progress):
    sys.stdout.write('\r%d%%' % progress)
    sys.stdout.flush()

def format_time(time):
    time_rounded = int(round((time)))
    return str(datetime.timedelta(seconds=time_rounded))

def train_model(model, train_data_loader, device=torch.device('cuda'), epochs=EPOCHS, lr=LEARNING_RATE, num_warmup_steps=WARMUP, eps=EPS):
    optimizer = AdamW(model.parameters(), lr=lr, eps=eps)

    checkpoint_path = './checkpoint.pt'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = AdamW(model.parameters(), lr=lr, eps=eps)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epochs -= checkpoint['epoch']

    num_training_steps = len(train_data_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    model.to(device)
    model.train()

    batch_size = train_data_loader.batch_size
    num_data = len(train_data_loader) * batch_size
    start_time = time.time()

    print(" --- training model")
    for epoch in range(epochs):
        total_loss = 0
        epoch_start_time = time.time()

        for step, batch in enumerate(train_data_loader):
            batch_inputs = tuple(t.to(device) for t in batch)
            inputs = {
                'input_ids': batch_inputs[0],
                'attention_mask': batch_inputs[1],
                'labels': batch_inputs[2]
            }

            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs[0]
            total_loss += loss.item()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            update_progress((step+1)*batch_size / num_data * 100)

        avg_train_loss = total_loss / len(train_data_loader)
        print(f' {epoch+1}/{epochs} - elapsed: {format_time(time.time() - epoch_start_time)}, average train loss: {avg_train_loss}')

        if epoch + 1 < epochs:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    print(f' --- train finished, elapsed: {format_time(time.time() - start_time)}')

def evaluate_model(model, test_data_loader, device=torch.device('cuda')):
    total_eval_loss = 0
    labels = np.array([])
    predictions = np.array([])

    model.to(device)
    model.eval()

    batch_size = test_data_loader.batch_size
    num_data = len(test_data_loader) * batch_size
    start_time = time.time()

    for step, batch in enumerate(test_data_loader):
        batch_inputs = tuple(t.to(device) for t in batch)
        inputs = {
            'input_ids': batch_inputs[0],
            'attention_mask': batch_inputs[1],
            'labels': batch_inputs[2]
        }

        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs[0]
            logits = outputs[1]

        total_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        labels = np.concatenate((batch_inputs[2].to('cpu').numpy(), labels))
        predictions = np.concatenate((logits.argmax(axis=1), predictions))

        update_progress((step+1) * batch_size / num_data * 100)
    
    avg_eval_loss = total_eval_loss / len(test_data_loader)
    print(f' f1: {f1_score(labels, predictions)}, evaluating loss: {avg_eval_loss:.4f}')
    print(f' {np.sum(predictions == labels)} / {predictions.shape[0]} ')
    print(f' --- evaluation finished {format_time(time.time() - start_time)}')


def draw_tensor_3D(tensor, x_label="X", y_label="Y", scale=5):
    data = tensor.numpy()

    if data.shape[0] == 1:
        data = data.squeeze(0)
    
    shape = data.shape
    x = np.arange(0, shape[1], 1)
    y = np.arange(0, shape[0], 1)
    x, y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, data, cmap='viridis')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel('value')

    max = np.max(data)
    if scale < max:
        scale = max
    ax.set_zlim(0, scale)

    plt.show()

def draw_activation(tensor, scale = 5):
    tensor = tensor.to('cpu').detach()
    tensor = torch.abs(tensor)
    draw_tensor_3D(tensor, "Channel", "Token", scale)

def draw_weight(tensor, scale = 5):
    tensor = tensor.to('cpu').detach()
    tensor = torch.abs(tensor)
    draw_tensor_3D(tensor, "Out Channel", "In Channel", scale)


def replace_modules(model, from_class, to_class):
    for name, module in model.named_children():
        if isinstance(module, from_class):
            target_object = to_class(module)
            setattr(model, name, target_object)
        else:
            replace_modules(module, from_class, to_class)