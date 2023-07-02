import datetime
import time
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset, RandomSampler

from copy import deepcopy


from sklearn.metrics import f1_score
from transformers import AdamW


DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_EPS = 1e-8

class CorrectedEncoder(nn.Module):
    def __init__(self, encoder, correction_model):
        super(CorrectedEncoder, self).__init__()
        self.encoder = encoder
        self.correction_model = correction_model

    def forward(
        self,
        embedding_output,
        attention_mask,
        head_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        past_key_values,
        use_cache,
        output_attentions,
        output_hidden_states,
        return_dict
    ):
        
        output = self.encoder(
            embedding_output,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_values,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

        corr = self.correction_model(embedding_output.unsqueeze(1))
        output['last_hidden_state'] += corr

        return output


# hh:mm:ss
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))    
    return str(datetime.timedelta(seconds=elapsed_rounded))


# o_model = original model, q_model = quantized(encoder only) model
def train_correction_model(corr_model, dataset_path, save_path = None, batch_size = DEFAULT_BATCH_SIZE, lr = DEFAULT_LEARNING_RATE, eps = DEFAULT_EPS, epochs = DEFAULT_EPOCHS):
    cuda = torch.device('cuda')

    corr_model.train()
    corr_model.to(cuda)

    data = torch.load(dataset_path)
    inputs = torch.tensor(data['inputs'])
    labels = torch.tensor(data['labels'])
    dataset = TensorDataset(inputs, labels)
    random_sampler = RandomSampler(dataset)
    data_loader = DataLoader(dataset, sampler=random_sampler, batch_size = batch_size)

    loss_fn = nn.MSELoss()
    optimizer = AdamW(corr_model.parameters(), lr = lr, eps = eps)
    start_time = time.time()

    def checkpoint(epoch, total_loss):
        torch.save({
            # 'epoch': epoch,
            'model_state_dict': corr_model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            # 'total_loss': total_loss
        }, save_path)

    for cur_epoch in range(epochs):
        total_loss = 0
        cur_time = time.time()
        print(f'--- training {cur_epoch + 1} / {epochs}')

        for step, batch in enumerate(data_loader):
            if step > 150:
                break

            optimizer.zero_grad()
            batch_inputs = tuple(t.to(cuda) for t in batch)
            batch_inputs = batch_inputs[0]
            batch_labels = batch_inputs[1]
            
            predictions = corr_model(batch_inputs.unsqueeze(1))

            loss = loss_fn(predictions, batch_labels)
            loss.backward()
            total_loss += loss.item()

            optimizer.step()
            print(loss)
        
        avg_train_loss = total_loss / len(dataset)
        print(f'average training loss : {avg_train_loss}, elapsed : {format_time(time.time() - cur_time)}')

        if save_path != None:
            checkpoint(cur_epoch, total_loss)

    print(f'--- train finished. elapsed : {format_time(time.time() - start_time)}')


def create_corrected_encoder(model, correction_model):
    c_model = deepcopy(model)
    c_model.bert.encoder = CorrectedEncoder(model.bert.encoder, correction_model)
    return c_model



def create_train_dataset(o_model, q_model, dataset, save_path = "./correction_data.pt"):
    cpu = torch.device('cpu')

    o_model.eval()
    o_model.to(cpu)
    q_model.eval()
    q_model.to(cpu)

    embedding_layer = o_model.bert.embeddings
    o_encoder = o_model.bert.encoder
    q_encoder = q_model.bert.encoder

    embedding_outputs = np.array([])
    labels = np.array([])
    tensor_size = 0

    for step, batch in enumerate(dataset):

        batch_inputs = tuple(t.to(cpu) for t in batch)
        batch_inputs = {
            'input_ids': batch_inputs[0],
        }

        with torch.no_grad():
            embedding_output = embedding_layer(**batch_inputs)
            o_encoder_output = o_encoder(embedding_output)
            q_encoder_output = q_encoder(embedding_output)
            label = o_encoder_output[0] - q_encoder_output[0]

            if labels.shape[0] == 0:
                embedding_outputs = embedding_output
                labels = label
                tensor_size = embedding_output.numel() * embedding_output.element_size() + label.numel() * label.element_size()
            else:

                output_size = embedding_output.numel() * embedding_output.element_size() + label.numel() * label.element_size()
                if tensor_size + output_size >= 4 * (1024 ** 3):
                    break

                tensor_size += output_size
                embedding_outputs = np.concatenate((embedding_outputs, embedding_output),axis=0)
                labels = np.concatenate((labels, label),axis=0)
    
    print(embedding_outputs.shape)
            

    torch.save({
        "inputs": embedding_outputs,
        "labels": labels
    }, "./train_dataset")
