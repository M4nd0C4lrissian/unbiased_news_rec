import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from torch.utils.data import DataLoader

from my_work.custom_article_embedding_dataset import CustomArticleEmbeddingDataset as CD

class Encoder(nn.Module):
    def __init__(self, bert_dim, intermediate_dim, final_dim, num_heads=8, dropout=0.1):
        super(Encoder, self).__init__()
        
        # Multi-Head Attention Layer
        self.self_attention = nn.MultiheadAttention(embed_dim=bert_dim, num_heads=num_heads, dropout=dropout)
        
        # Feed-forward layers
        self.fcj = nn.Linear(bert_dim, intermediate_dim)
        self.fc1 = nn.Linear(intermediate_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim + intermediate_dim, intermediate_dim)
        self.fc3 = nn.Linear(intermediate_dim + intermediate_dim, final_dim)  # Residual connection
        
        # Final activation - remove this?
        self.final_activation = nn.Sigmoid()
        
        # Layer normalization for residual connections
        self.layer_norm = nn.LayerNorm(bert_dim)

    def forward(self, x):
        # x shape: (batch_size, T, bert_dim)
        
        # Prepare input for attention: need to transpose to (T, batch_size, bert_dim)
        x = x.transpose(0, 1)  # (T, batch_size, bert_dim)
        x = x.transpose(0,2)
        # Apply multi-head self-attention
        attn_output, _ = self.self_attention(x, x, x)  # (T, batch_size, bert_dim)
        
        # Add residual connection and normalize
        x = self.layer_norm(attn_output + x)
        
        # Pooling to aggregate sequence info (mean pooling)
        x = torch.mean(x, dim=0)  # (batch_size, bert_dim)
        
        # Fully connected layers with residual connections
        cj = self.final_activation(self.fcj(x))  
        x1 = F.leaky_relu(self.fc1(cj), negative_slope=0.1)
        x2 = F.leaky_relu(self.fc2(torch.cat((cj, x1), dim=-1)), negative_slope=0.1)
        x3 = F.leaky_relu(self.fc3(torch.cat((cj, x2), dim=-1)), negative_slope=0.1)  # (batch_size, final_dim)
        ##x4 = self.final_activation(x3)  # (batch_size, final_dim)
        
        return x3, cj

# Decoder with residual connections
class Decoder(nn.Module):
    def __init__(self, input_dim, intermediate_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim + input_dim, intermediate_dim)
        self.fc3 = nn.Linear(intermediate_dim + input_dim, output_dim)
        
        nn.init.uniform_(self.fc1.weight, a=-2.0, b=2.0)
        nn.init.uniform_(self.fc2.weight, a=-2.0, b=2.0)
        nn.init.uniform_(self.fc3.weight, a=-2.0, b=2.0)

    def forward(self, x):
        x1 = F.leaky_relu(self.fc1(x), negative_slope=0.1)
        x2 = F.leaky_relu(self.fc2(torch.cat((x1, x), dim=-1)), negative_slope=0.1)
        x3 = F.leaky_relu(self.fc3(torch.cat((x2, x), dim=-1)), negative_slope=0.1)
        return x3

# Polarity classifier
class PolarityClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(PolarityClassifier, self).__init__()
        self.f1 = nn.Linear(input_dim, input_dim)
        self.f2 = nn.Linear(input_dim, input_dim)
        self.fc = nn.Linear(input_dim, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x1 = F.leaky_relu(self.f1(x))
        x2 = F.leaky_relu(self.f2(x1))
        # return self.softmax(self.fc(x2))
        x3 = self.fc(x2)
        
        return x3

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

        return input, encoded, polarity_decoded, polarity_free_decoded, polarity_pred, polarity_free_pred


def diversity_loss(embeddings):
    normed_embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    cosine_sim = torch.mm(normed_embeddings, normed_embeddings.T)
    loss = torch.mean(cosine_sim) - torch.diag(cosine_sim).mean()
    return loss

def orthogonality_loss_with_labels(embeddings, labels, num_classes, target_norm=1.0, weight_magnitude=0.1):
    """
    Computes orthogonality loss across embeddings grouped by labels, with normalization and magnitude regularization.
    
    Args:
    - embeddings: Tensor of shape (batch_size, embedding_dim).
    - labels: Tensor of shape (batch_size,) containing topic labels (0, 1, ..., num_classes - 1).
    - num_classes: Integer specifying the number of topic classes.
    - target_norm: Target magnitude for embeddings (default: 1.0).
    - weight_magnitude: Weight for the magnitude regularization term (default: 0.1).
    
    Returns:
    - loss: Scalar tensor, the computed total loss.
    """
    loss = 0.0
    device = embeddings.device
    
    # Normalize embeddings to unit length
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Split embeddings by class
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            # Get embeddings for class i and class j
            embeddings_i = normalized_embeddings[labels == i]
            embeddings_j = normalized_embeddings[labels == j]
            
            if embeddings_i.size(0) == 0 or embeddings_j.size(0) == 0:
                # Skip if there are no embeddings for a class
                continue
            
            # Compute cross-orthogonality loss
            cross_gram_matrix = torch.mm(embeddings_i, embeddings_j.T)  # Shape: (num_i, num_j)
            loss += (cross_gram_matrix ** 2).sum()  # Encourage dot products to be near zero
    
    # Add magnitude regularization term
    norms = torch.norm(embeddings, dim=1)
    magnitude_loss = ((norms - target_norm) ** 2).mean()
    
    # Total loss: orthogonality loss + magnitude regularization
    total_loss = loss + weight_magnitude * magnitude_loss
    return total_loss

# Losses
def compute_losses(c_j, encoder_output, polarity_decoded, polarity_free_decoded, polarity_pred, polarity_free_pred, polarity_labels):
    # Reconstruction loss
    reconstruction_loss = 0.5 * F.mse_loss(
        torch.cat((polarity_decoded, polarity_free_decoded), dim=-1), c_j
    )
    
    lambda_diversity = 0.1
    embedding_diversity_loss = lambda_diversity * diversity_loss(torch.cat((polarity_decoded, polarity_free_decoded), dim=0))
    
    ortho_lambda_diversity = 0.00001
    encoded_orthogonality_loss = 1

    # Classification loss
    logits = polarity_pred.float()
    targets = polarity_labels.long()
    criterion = nn.CrossEntropyLoss()
    classification_loss = criterion(logits, targets)
    ##classification_loss = F.binary_cross_entropy_with_logits(polarity_pred.float(), polarity_labels.float())

    # confusion_loss = 1.0 / (F.binary_cross_entropy_with_logits(polarity_free_pred.float(), polarity_labels.float()) + 1e-8)
    ##confusion_loss = - F.binary_cross_entropy_with_logits(polarity_free_pred.float(), polarity_labels.float())
    
    logits = polarity_free_pred.float()
    confusion_loss = 1 / criterion(logits, targets)


    return reconstruction_loss, classification_loss, confusion_loss, embedding_diversity_loss, encoded_orthogonality_loss

# Training loop
def train(model, scheduler, dataset, val_dataset, optimizer, path_to_data, embedding_batch_num, num_epochs=10):

    data_loader = DataLoader(dataset, batch_size= embedding_batch_num // 20, shuffle=True)
    ##val_loader = DataLoader(val_dataset, batch_size=embedding_batch_num // 20, shuffle=False)
    
    lr_slowdown = 0
    patience = 0
    tolerance = 3
    
    best_train_loss = None
    
    
    for epoch in range(num_epochs):

        dataset.update_dataset(path_to_data + f'text_embedding_{epoch}.pt', path_to_data + f'title_embedding_{epoch}.pt', [epoch*embedding_batch_num, (epoch + 1)*embedding_batch_num])
        
        # Training phase
        model.train()
        total_train_loss = 0

        # if lr_slowdown < 6:
        #     if epoch % 4 == 0 and epoch > 0:
        #         for g in optimizer.param_groups:
        #             g['lr'] = g['lr'] / 10
        #         lr_slowdown += 1
                
        # if epoch == 10:
        #     for g in optimizer.param_groups:
        #         g['lr'] = 0.1
        
        for batch_idx, (bert1, bert2, polarity_labels) in enumerate(data_loader):
            optimizer.zero_grad()

            # Forward pass
            c_j, encoded, polarity_decoded, polarity_free_decoded, polarity_pred, polarity_free_pred = model(bert1, bert2)
            

            # Loss computation
            recon_loss, class_loss, conf_loss, embedding_diversity_loss, encoded_orthogonality_loss = compute_losses(
                c_j, encoded, polarity_decoded, polarity_free_decoded, polarity_pred, polarity_free_pred, polarity_labels
            )

            ##increase weight of confusion loss?
            total_loss = recon_loss + class_loss + conf_loss
            
            total_train_loss += total_loss.item()
            print(f"Reconstruction loss: {recon_loss}")
            print(f"Classification loss: {class_loss}")
            print(f"Confusion loss: {conf_loss}")

            # Backward pass
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(data_loader)}, Batch Loss: {total_loss.item()}")

        avg_train_loss = total_train_loss / len(data_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Avg Training Loss: {avg_train_loss}")
        
        if best_train_loss:
            if avg_train_loss > best_train_loss:
                patience += 1
            
            else:
                best_train_loss = avg_train_loss
                patience = 0
                
        else:
            best_train_loss = avg_train_loss 
            patience = 0
            
        if patience > tolerance:
            print("early stopping")
            break
        
        # # Validation phase
        # model.eval()
        # total_val_loss = 0
        # with torch.no_grad():
        #     for bert1, bert2, polarity_labels in val_loader:
        #         # Forward pass
        #         encoded, polarity_decoded, polarity_free_decoded, polarity_pred, polarity_free_pred = model(bert1, bert2)

        #         # Loss computation
        #         recon_loss, class_loss, conf_loss = compute_losses(
        #             encoded, polarity_decoded, polarity_free_decoded, polarity_pred, polarity_free_pred, polarity_labels
        #         )
        #         total_loss = recon_loss + class_loss + conf_loss
        #         total_val_loss += total_loss.item()
        #         # print(f"Reconstruction loss: {recon_loss}")
        #         # print(f"Classification loss: {class_loss}")
        #         # print(f"Confusion loss: {conf_loss}")

        # avg_val_loss = total_val_loss / len(val_loader)
        # print(f"Epoch {epoch + 1}/{num_epochs}, Avg Validation Loss: {avg_val_loss}")

if __name__ == '__main__':
    # Example usage
    bert_dim = 768  # Example BERT embedding size
    intermediate_dim = 256
    encoder_output_dim = 128
    num_classes = 3  # {-1, 0, 1}
    
    
    
    encoder = Encoder(bert_dim, intermediate_dim, encoder_output_dim)
    epochs = 3
    
    embedding_batch_num = 1000
    
    labels_file = "src\data\\auto_encoder_training\\training_data\\partisan_labels.csv"
    
    path_to_data = "D:\Bert-Embeddings\\training_data\\"
    
    text_embedding_file = path_to_data + "text_embedding_0.pt"
    title_embedding_file = path_to_data + "title_embedding_0.pt"

    dataset = CD(labels_file, [text_embedding_file], [title_embedding_file], [0, embedding_batch_num])
    
    data_loader = DataLoader(dataset, batch_size= embedding_batch_num // 20, shuffle=True)
    optimizer = optim.Adam(encoder.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (bert1, bert2, polarity_labels) in enumerate(data_loader):
            # inputs, labels = inputs.to(device), labels.to(device)
            
            concatenated = torch.cat((bert1, bert2), dim=-1)
            # Forward pass
            embeddings, c_j = encoder(concatenated)
            
            # Compute the combined loss
            loss = orthogonality_loss_with_labels(c_j, polarity_labels, 3, weight_magnitude=0.1, target_norm=1.0)
            
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
    
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")
    
    for param in encoder.self_attention.parameters():
        print('Here 1')
        param.requires_grad = False
    
    for param in encoder.fcj.parameters():
        print('Here 2')
        param.requires_grad = False

    model = DualDecoderModel(bert_dim, intermediate_dim, encoder_output_dim, num_classes)
    
    model.encoder = encoder
    
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=0.001
    )


    # val_path = "D:\Bert-Embeddings\\validation_data\\"
    
    
    # labels_file = "src\data\\auto_encoder_training\\validation_data\\validation_partisan_labels.csv"
    # text_paths = []
    # title_paths = []
    # for i in range(4):
    #     text_paths.append(val_path + f"text_embedding_{i}.pt")
    #     title_paths.append(val_path + f"title_embedding_{i}.pt")
    
    # val_dataset = CD(labels_file, text_paths, title_paths, [0, 4000])
    
    warmup_steps = 6000  # Number of steps to warm up
    total_steps = 32000  # Total training steps
    decay_rate = 0.95    # Exponential decay rate

    # Define the learning rate schedule
    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps  # Linear warmup
        else:
            return decay_rate ** ((step - warmup_steps) / (total_steps - warmup_steps))  # Exponential decay

    # Scheduler
    scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule)

    val_dataset = None
    
    # Assuming `data_loader` is a PyTorch DataLoader with batches of (bert1, bert2, polarity_labels)
    train(model, scheduler, dataset, val_dataset, optimizer, path_to_data, embedding_batch_num, num_epochs=32)
    torch.save(model.state_dict, 'src\my_work\models\\full_model.pt')
    
    
    polarity_decoder = model.polarity_decoder
    polarity_free_decoder = model.polarity_free_decoder
    encoder = model.encoder
    
    torch.save(polarity_decoder.state_dict(), 'src\my_work\models\polarity_decoder.pt')
    torch.save(polarity_free_decoder.state_dict(), 'src\my_work\models\polarity_free_decoder.pt')
    torch.save(encoder.state_dict(), 'src\my_work\models\encoder.pt')