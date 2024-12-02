import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

from custom_article_embedding_dataset import CustomArticleEmbeddingDataset as CD

# Encoder with residual connections
class Encoder(nn.Module):
    def __init__(self, intermediate_dim, final_dim):
        super(Encoder, self).__init__()
        ##techincally missing one of the layers from the paper
        self.fc1 = nn.Linear(intermediate_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim * 2, final_dim)  # Residual connection

    def forward(self, x):
        # # x shape: (batch_size, T, 2 * bert_dim)
        # # Transpose for Conv1d: (batch_size, 2 * bert_dim, T)
        # x = x.transpose(1, 2)
        
        # # Apply Conv1d
        # x = self.conv1d(x)  # (batch_size, intermediate_dim, T)
        # x = self.activation(x)
        
        # Pooling to aggregate sequence info (mean pooling)
        # x = torch.mean(x, dim=-1)  # (batch_size, intermediate_dim)
        
        # Fully connected layers with residual connections
        x1 = F.leaky_relu(self.fc1(x), negative_slope=0.1)  # (batch_size, intermediate_dim)
        x2 = F.leaky_relu(self.fc2(torch.cat((x, x1), dim=-1)), negative_slope=0.1)  # (batch_size, final_dim)
    
        return x2

class DenseProjector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DenseProjector, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten 768 x 188 into a single dimension
        return self.fc(x)


# Decoder with residual connections
class Decoder(nn.Module):
    def __init__(self, input_dim, intermediate_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.fc3 = nn.Linear(intermediate_dim * 2, output_dim)

    def forward(self, x):
        x1 = F.leaky_relu(self.fc1(x), negative_slope=0.1)
        x2 = F.leaky_relu(self.fc2(x1), negative_slope=0.1)
        x3 = F.leaky_relu(self.fc3(torch.cat((x1, x2), dim=-1)), negative_slope=0.1)
        return x3

# Polarity classifier
class PolarityClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(PolarityClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.fc(x))

# Full Model
class DualDecoderModel(nn.Module):
    def __init__(self, bert_dim, intermediate_dim, encoder_output_dim, num_classes):
        super(DualDecoderModel, self).__init__()

        self.projector = DenseProjector(768 * 188, intermediate_dim)
        self.encoder = Encoder(intermediate_dim, encoder_output_dim)
        self.polarity_decoder = Decoder(encoder_output_dim, intermediate_dim, encoder_output_dim)
        self.polarity_free_decoder = Decoder(encoder_output_dim, intermediate_dim, encoder_output_dim)
        self.polarity_classifier = PolarityClassifier(encoder_output_dim, num_classes)

    def forward(self, bert1, bert2):
        # Concatenate BERT embeddings
        concatenated = torch.cat((bert1, bert2), dim=-1)
        
        projected = self.projector(concatenated)

        # Encode
        encoded = self.encoder(projected)

        # Decode
        polarity_decoded = self.polarity_decoder(encoded)
        polarity_free_decoded = self.polarity_free_decoder(encoded)

        # Classify polarity
        polarity_pred = self.polarity_classifier(polarity_decoded)
        polarity_free_pred = self.polarity_classifier(polarity_free_decoded)

        return projected, encoded, polarity_decoded, polarity_free_decoded, polarity_pred, polarity_free_pred

# Losses
def compute_losses(original_embedding, polarity_decoded, polarity_free_decoded, polarity_pred, polarity_free_pred, polarity_labels):
    # Reconstruction loss
    reconstruction_loss = F.mse_loss(
        torch.cat((polarity_decoded, polarity_free_decoded), dim=-1), original_embedding
    )

    # Classification loss
    classification_loss = F.binary_cross_entropy_with_logits(polarity_pred.float(), polarity_labels.float())

    #this is also wrong
    confusion_loss = 1.0 / (F.binary_cross_entropy_with_logits(polarity_free_pred.float(), polarity_labels.float()) + 1e-8)

    return reconstruction_loss, classification_loss, confusion_loss

# Training loop
def train(model, dataset, optimizer, path_to_data, embedding_batch_num, num_epochs=10):

    data_loader = DataLoader(dataset, batch_size= embedding_batch_num // 10, shuffle=True)

    for epoch in range(num_epochs):

        dataset.update_dataset(path_to_data + f'text_embedding_{epoch}.pt', path_to_data + f'title_embedding_{epoch}.pt', [epoch*embedding_batch_num, (epoch + 1)*embedding_batch_num])

        for bert1, bert2, polarity_labels in data_loader:
            optimizer.zero_grad()

            # Forward pass
            projected, encoded, polarity_decoded, polarity_free_decoded, polarity_pred, polarity_free_pred = model(bert1, bert2)

            # Loss computation
            recon_loss, class_loss, conf_loss = compute_losses(
                projected, polarity_decoded, polarity_free_decoded, polarity_pred, polarity_free_pred, polarity_labels
            )

            total_loss = recon_loss + class_loss + conf_loss
            total_loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss.item()}")

if __name__ == '__main__':

    bert_dim = 60 + 128 # Concatenated token limits of titles and text
    intermediate_dim = 128
    encoder_output_dim = 64
    num_classes = 3  # {-1, 0, 1}

    model = DualDecoderModel(bert_dim, intermediate_dim, encoder_output_dim, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    path_to_data = "unbiased_news_rec\src\data\\auto_encoder_training\\"
    embedding_batch_num = 1000

    labels_file = path_to_data + "partisan_labels.csv"
    text_embedding_file = path_to_data + "text_embedding_0.pt"
    title_embedding_file = path_to_data + "title_embedding_0.pt"

    dataset = CD(labels_file, text_embedding_file, title_embedding_file, [0, embedding_batch_num])

    # Assuming `data_loader` is a PyTorch DataLoader with batches of (bert1, bert2, polarity_labels)
    train(model, dataset, optimizer, path_to_data, embedding_batch_num, num_epochs=1)

    torch.save(model.state_dict(), 'unbiased_news_rec\\src\\data\\auto_encoder_training\\test_state_dict.pt')