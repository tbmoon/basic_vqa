import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from data_loader import get_loader
from models import VqaModel
from torchvision import transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
        
    data_loader = get_loader(
        input_dir=args.input_dir,
        input_vqa='train.npy',
        max_qst_length=args.max_qst_length,
        transform=transform,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)

    qst_vocab_size = data_loader.dataset.vocab_qst.num_vocab
    ans_vocab_size = data_loader.dataset.vocab_ans.num_vocab
    
    model = VqaModel(
        embed_size=args.embed_size,
        qst_vocab_size=qst_vocab_size,
        ans_vocab_size=ans_vocab_size,
        word_embed_size=args.word_embed_size,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    params = list(model.parameters()) # TO DO: UPDATE THIS
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    for iepoch in range(args.num_epochs):
        
        loss_sum = 0.0
        
        for isample, sample in enumerate(data_loader):
                
            image = sample['image'].to(device)
            question = sample['question'].to(device)
            answer = sample['answer'].to(device)
            
            prediction = model(image, question)
            loss = criterion(prediction, answer)
            
            model.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_sum += loss.item()

        avg_loss = loss_sum / len(data_loader.dataset)
        print('Epoch [{}/{}] Loss: {:.8f}'.format(iepoch+1, args.num_epochs, avg_loss))    
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./datasets',
                        help='input directory for visual question answering.')
    parser.add_argument('--max_qst_length', type=int, default=30,
                        help='maximum length of question. the length in the VQA dataset = 26.')
    parser.add_argument('--embed_size', type=int, default=1024,
                        help='embedding size of feature vector for both image and question.')
    parser.add_argument('--word_embed_size', type=int, default=300,
                        help='embedding size of word used for the input in RNN(LSTM).')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers of RNN(LSTM).')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='hidden_size.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate for training')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='number of epochs.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch_size.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of processes working on cpu.')
    args = parser.parse_args()
    
    main(args)