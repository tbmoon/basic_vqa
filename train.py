import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from data_loader import get_loader
from models import VqaModel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    f = open(os.path.join(args.log_dir, 'log.txt'), 'w')
    
    data_loader = get_loader(
        input_dir=args.input_dir,
        input_vqa_train='train.npy',
        input_vqa_valid='valid.npy',
        max_qst_length=args.max_qst_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    qst_vocab_size = data_loader['train'].dataset.qst_vocab.vocab_size
    ans_vocab_size = data_loader['train'].dataset.ans_vocab.vocab_size

    model = VqaModel(
        embed_size=args.embed_size,
        qst_vocab_size=qst_vocab_size,
        ans_vocab_size=ans_vocab_size,
        word_embed_size=args.word_embed_size,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size).to(device)

    criterion = nn.CrossEntropyLoss()

    params = list(model.img_encoder.fc.parameters()) \
        + list(model.qst_encoder.parameters()) \
        + list(model.fc.parameters())

    optimizer = optim.Adam(params, lr=args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    print('\n')

    for epoch in range(args.num_epochs):
        
        for phase in ['train', 'valid']:
            
            loss_sum = 0.0
            batch_step_size = len(data_loader[phase].dataset) / args.batch_size

            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            for batch_idx, batch_sample in enumerate(data_loader[phase]):
                    
                image = batch_sample['image'].to(device)
                question = batch_sample['question'].to(device)
                answer = batch_sample['answer'].to(device)

                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    
                    prediction = model(image, question)
                    loss = criterion(prediction, answer)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                loss_sum += loss.item()

                # Print the loss in a mini-batch.
                if batch_idx % 100 == 0:
                    print('| {} SET | Epoch [{:02d}/{:02d}], Step [{:04d}/{:04d}], Loss: {:.8f}'
                          .format(phase.upper(), epoch+1, args.num_epochs, batch_idx, int(batch_step_size), loss.item()))

            # Print the average loss (loss per mini-batch).
            avg_loss = loss_sum / batch_step_size
            print('| {} SET | Epoch [{:02d}/{:02d}], Avg. Loss: {:.8f} \n'
                  .format(phase.upper(), epoch+1, args.num_epochs, avg_loss))
            
            # Log the average loss as an epoch.
            f.write(phase + '\t' + str(epoch+1) + '\t' + str(avg_loss) + '\n')

            # Save the model check points.
            if (epoch+1) % args.save_step == 0:
                torch.save({'epoch': epoch, 'state_dict': model.state_dict()},
                           os.path.join(args.model_dir, 'model-epoch-{}.ckpt'.format(epoch+1)))

    f.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='./datasets',
                        help='input directory for visual question answering.')

    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='directory for logs.')

    parser.add_argument('--model_dir', type=str, default='./models',
                        help='directory for saved models.')

    parser.add_argument('--max_qst_length', type=int, default=30,
                        help='maximum length of question. \
                              the length in the VQA dataset = 26.')

    parser.add_argument('--embed_size', type=int, default=1024,
                        help='embedding size of feature vector \
                              for both image and question.')

    parser.add_argument('--word_embed_size', type=int, default=300,
                        help='embedding size of word \
                              used for the input in the LSTM.')

    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers of the RNN(LSTM).')

    parser.add_argument('--hidden_size', type=int, default=512,
                        help='hidden_size in the LSTM.')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate for training.')

    parser.add_argument('--step_size', type=int, default=5,
                        help='period of learning rate decay.')

    parser.add_argument('--gamma', type=float, default=0.1,
                        help='multiplicative factor of learning rate decay.')

    parser.add_argument('--num_epochs', type=int, default=20,
                        help='number of epochs.')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size.')

    parser.add_argument('--num_workers', type=int, default=16,
                        help='number of processes working on cpu.')

    parser.add_argument('--save_step', type=int, default=5,
                        help='save step of model.')

    args = parser.parse_args()

    main(args)
