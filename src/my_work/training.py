import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

from my_work.custom_article_embedding_dataset import CustomArticleEmbeddingDataset as CD

# Encoder with residual connections
## make model bigger
class Encoder(nn.Module):
    def __init__(self, bert_dim, intermediate_dim, final_dim):
        super(Encoder, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=768, out_channels=intermediate_dim, kernel_size=1)
        self.fc1 = nn.Linear(intermediate_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim * 2, final_dim)  # Residual connection
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, T, 2 * bert_dim)
        # Transpose for Conv1d: (batch_size, 2 * bert_dim, T)
        ##x = x.transpose(1, 2)
        
        # Apply Conv1d
        x = self.conv1d(x)  # (batch_size, intermediate_dim, T)
        x = F.relu(x)
        
        # Pooling to aggregate sequence info (mean pooling)
        x = torch.mean(x, dim=-1)  # (batch_size, intermediate_dim)
        
        # Fully connected layers with residual connections
        x1 = F.leaky_relu(self.fc1(x), negative_slope=0.1)  # (batch_size, intermediate_dim)
        x2 = F.leaky_relu(self.fc2(torch.cat((x, x1), dim=-1)), negative_slope=0.1)  # (batch_size, final_dim)
        x3 = self.final_activation(x2)  # (batch_size, final_dim)
        return x3, x

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
        concatenated_dim = bert_dim
        self.encoder = Encoder(concatenated_dim, intermediate_dim, encoder_output_dim)
        self.polarity_decoder = Decoder(encoder_output_dim, intermediate_dim, encoder_output_dim)
        self.polarity_free_decoder = Decoder(encoder_output_dim, intermediate_dim, encoder_output_dim)
        self.polarity_classifier = PolarityClassifier(encoder_output_dim, num_classes)

    def forward(self, bert1, bert2):
        # Concatenate BERT embeddings
        concatenated = torch.cat((bert1, bert2), dim=-1)

        # Encode
        encoded, input = self.encoder(concatenated)

        # Decode
        polarity_decoded = self.polarity_decoder(encoded)
        polarity_free_decoded = self.polarity_free_decoder(encoded)

        # Classify polarity
        polarity_pred = self.polarity_classifier(polarity_decoded)
        polarity_free_pred = self.polarity_classifier(polarity_free_decoded)

        return input, polarity_decoded, polarity_free_decoded, polarity_pred, polarity_free_pred

# Losses
def compute_losses(encoder_output, polarity_decoded, polarity_free_decoded, polarity_pred, polarity_free_pred, polarity_labels):
    # Reconstruction loss
    reconstruction_loss = F.mse_loss(
        torch.cat((polarity_decoded, polarity_free_decoded), dim=-1), encoder_output
    )

    # Classification loss
    classification_loss = F.binary_cross_entropy_with_logits(polarity_pred.float(), polarity_labels.float())

    confusion_loss = 1.0 / (F.binary_cross_entropy_with_logits(polarity_free_pred.float(), polarity_labels.float()) + 1e-8)

    return reconstruction_loss, classification_loss, confusion_loss

# Training loop
def train(model, dataset, val_dataset, optimizer, path_to_data, embedding_batch_num, num_epochs=10):

    data_loader = DataLoader(dataset, batch_size= embedding_batch_num // 20, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=embedding_batch_num // 20, shuffle=False)

    for epoch in range(num_epochs):

        dataset.update_dataset(path_to_data + f'text_embedding_{epoch}.pt', path_to_data + f'title_embedding_{epoch}.pt', [epoch*embedding_batch_num, (epoch + 1)*embedding_batch_num])
        
        # Training phase
        model.train()
        total_train_loss = 0

        for batch_idx, (bert1, bert2, polarity_labels) in enumerate(data_loader):
            optimizer.zero_grad()

            # Forward pass
            encoded, polarity_decoded, polarity_free_decoded, polarity_pred, polarity_free_pred = model(bert1, bert2)

            # Loss computation
            recon_loss, class_loss, conf_loss = compute_losses(
                encoded, polarity_decoded, polarity_free_decoded, polarity_pred, polarity_free_pred, polarity_labels
            )

            total_loss = recon_loss + class_loss + conf_loss
            total_train_loss += total_loss.item()
            # print(f"Reconstruction loss: {recon_loss}")
            # print(f"Classification loss: {class_loss}")
            # print(f"Confusion loss: {conf_loss}")

            # Backward pass
            total_loss.backward()
            optimizer.step()
            print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(data_loader)}, Batch Loss: {total_loss.item()}")

        avg_train_loss = total_train_loss / len(data_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Avg Training Loss: {avg_train_loss}")
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for bert1, bert2, polarity_labels in val_loader:
                # Forward pass
                encoded, polarity_decoded, polarity_free_decoded, polarity_pred, polarity_free_pred = model(bert1, bert2)

                # Loss computation
                recon_loss, class_loss, conf_loss = compute_losses(
                    encoded, polarity_decoded, polarity_free_decoded, polarity_pred, polarity_free_pred, polarity_labels
                )
                total_loss = recon_loss + class_loss + conf_loss
                total_val_loss += total_loss.item()
                # print(f"Reconstruction loss: {recon_loss}")
                # print(f"Classification loss: {class_loss}")
                # print(f"Confusion loss: {conf_loss}")

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Avg Validation Loss: {avg_val_loss}")

if __name__ == '__main__':
    # Example usage
    bert_dim = 60 + 256  # Example BERT embedding size
    intermediate_dim = 256
    encoder_output_dim = 128
    num_classes = 3  # {-1, 0, 1}

    model = DualDecoderModel(bert_dim, intermediate_dim, encoder_output_dim, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    embedding_batch_num = 1000

    labels_file = "src\data\\auto_encoder_training\\training_data\\partisan_labels.csv"
    
    path_to_data = "D:\Bert-Embeddings\\training_data\\"
    
    text_embedding_file = path_to_data + "text_embedding_0.pt"
    title_embedding_file = path_to_data + "title_embedding_0.pt"

    dataset = CD(labels_file, [text_embedding_file], [title_embedding_file], [0, embedding_batch_num])
    
    val_path = "D:\Bert-Embeddings\\validation_data\\"
    
    
    labels_file = "src\data\\auto_encoder_training\\validation_data\\validation_partisan_labels.csv"
    text_paths = []
    title_paths = []
    for i in range(4):
        text_paths.append(val_path + f"text_embedding_{i}.pt")
        title_paths.append(val_path + f"title_embedding_{i}.pt")
    
    val_dataset = CD(labels_file, text_paths, title_paths, [0, 4000])

    # Assuming `data_loader` is a PyTorch DataLoader with batches of (bert1, bert2, polarity_labels)
    train(model, dataset, val_dataset, optimizer, path_to_data, embedding_batch_num, num_epochs=10)
    torch.save(model.state_dict, 'src\my_work\models\\full_model.pt')
    
    
    polarity_decoder = model.polarity_decoder
    polarity_free_decoder = model.polarity_free_decoder
    encoder = model.encoder
    
    torch.save(polarity_decoder.state_dict(), 'src\my_work\models\polarity_decoder.pt')
    torch.save(polarity_free_decoder.state_dict(), 'src\my_work\models\polarity_free_decoder.pt')
    torch.save(encoder.state_dict(), 'src\my_work\models\encoder.pt')