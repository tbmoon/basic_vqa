import torch
import torch.nn as nn
import torchvision.models as models


class ImgEncoder(nn.Module):
    
    def __init__(self, embed_size):

        """
        (1) Load the pretrained model as you want.
            cf) one needs to check structure of model using 'print(model)' to remove last fc layer from the model. 
        (2) Replace final fc layer (score values from the ImageNet) with new fc layer (image feature).
        (3) Normalize feature vector.
        """
        super(ImgEncoder, self).__init__()
        model = models.vgg16(pretrained=True)
        in_features = model.classifier[-1].in_features                            # input size of feature vector 
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1]) # remove last fc layer
        
        self.model = model                               # loaded model without last fc layer
        self.linear = nn.Linear(in_features, embed_size) # feature vector of image
        
        
    def forward(self, image):
        
        """
        Extract feature vector from image vector.
        """
        with torch.no_grad():
            img_feature = self.model(image)    # [batch_size, vgg16_fc=4096]
        img_feature = self.linear(img_feature) # [batch_size, embed_size]

        return img_feature

    
class QstEncoder(nn.Module):
    
    def __init__(self, qst_word_size, word_embed_size, embed_size, num_layers, hidden_size):
        
        super(QstEncoder, self).__init__()        
        self.word2vec = nn.Embedding(qst_word_size, word_embed_size) 
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(2*num_layers*hidden_size, embed_size) # 2 for hidden and cell states
        
        
    def forward(self, question):
        
        qst_vec = self.word2vec(question)                             # [batch_size, max_qst_length=30, word_embed_size=300]
        qst_vec = self.tanh(qst_vec)
        qst_vec = qst_vec.transpose(0, 1)                             # [max_qst_length=30, batch_size, word_embed_size=300]
        _, (hidden, cell) = self.lstm(qst_vec)                        # [num_layers=2, batch_size, hidden_size=512]
        qst_feature = torch.cat((hidden, cell), 2)                    # [num_layers=2, batch_size, 2*hidden_size=1024]
        qst_feature = qst_feature.transpose(0, 1)                     # [batch_size, num_layers=2, 2*hidden_size=1024]
        qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size=2048]        
        qst_feature = self.linear(qst_feature)                        # [batch_size, embed_size]

        return qst_feature
    
    
class VqaModel(nn.Module):
    
    def __init__(self, embed_size, qst_word_size, word_embed_size, num_layers, hidden_size):
        
        super(VqaModel, self).__init__()
        self.imgEncoder = ImgEncoder(embed_size)
        self.qstEncoder = QstEncoder(qst_word_size, word_embed_size, embed_size, num_layers, hidden_size)
        
        
    def forward(self, img, qst):
        
        img_feature = self.imgEncoder(img)
        qst_feature = self.qstEncoder(qst)
        
        return img_feature
        

    




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
