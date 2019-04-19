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
        
    data_loader = get_loader(input_dir=args.input_dir,
                             input_vqa='train.npy',
                             max_qst_length=args.max_qst_length,
                             transform=transform,
                             batch_size=args.batch_size,
                             shuffle=False, # True
                             num_workers=args.num_workers)

    qst_word_size = data_loader.dataset.vocab_qst.num_vocab
    
    model = VqaModel(args.embed_size, qst_word_size, args.word_embed_size, args.num_layers, args.hidden_size)

    criterion = nn.CrossEntropyLoss()
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    for iepoch in range(args.num_epochs):
        
        for isample, sample in enumerate(data_loader):
            
            if isample == 1:
                break
                
            image = sample['image']
            question = sample['question']
            answer = sample['answer']
            
            #print(question)
            #print(question.shape)
            prediction = model(image, question)
    
    
            # Forward, backward and optimize
            #features = encoder(images)
            #outputs = decoder(features, captions, lengths)
            #loss = criterion(outputs, targets)
            #decoder.zero_grad()
            #encoder.zero_grad()
            #loss.backward()
            #optimizer.step()
    
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./datasets',
                        help='input directory for visual question answering.')
    parser.add_argument('--max_qst_length', type=int, default=30,
                        help='maximum length of question. the length in the VQA dataset = 26.')
    parser.add_argument('--embed_size', type=int, default=5, #1024
                        help='embedding size of feature vector for both image and question.')
    parser.add_argument('--word_embed_size', type=int, default=300,
                        help='embedding size of word used for the input in RNN(LSTM).')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers of RNN(LSTM).')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='hidden_size.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate for training')
    parser.add_argument('--num_epochs', type=int, default=1, # 100
                        help='number of epochs.')
    parser.add_argument('--batch_size', type=int, default=3, # 8
                        help='batch_size.')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of processes working on cpu.')
    args = parser.parse_args()
    
    main(args)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
