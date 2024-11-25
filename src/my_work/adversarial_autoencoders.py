import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        embedding = self.fc2(x)
        return embedding

# Decoder
class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        reconstructed = self.fc2(x)
        return reconstructed

# Polarity Classifier
class PolarityClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(PolarityClassifier, self).__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

# Full Model
class DualDecoderModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, output_dim, num_classes):
        super(DualDecoderModel, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, embedding_dim)
        self.polarity_decoder = Decoder(embedding_dim, hidden_dim, output_dim)
        self.polarity_free_decoder = Decoder(embedding_dim, hidden_dim, output_dim)
        self.polarity_classifier = PolarityClassifier(output_dim, num_classes)

    def forward(self, x):
        embedding = self.encoder(x)
        polarity_reconstruction = self.polarity_decoder(embedding)
        polarity_free_reconstruction = self.polarity_free_decoder(embedding)
        polarity_pred = self.polarity_classifier(polarity_reconstruction)
        return embedding, polarity_reconstruction, polarity_free_reconstruction, polarity_pred

# Losses
def compute_losses(x, polarity_recon, polarity_free_recon, polarity_pred, polarity_labels, polarity_classifier):
    # Reconstruction loss
    recon_loss = F.mse_loss(polarity_recon, x) + F.mse_loss(polarity_free_recon, x)
    
    # Polarity prediction loss
    polarity_loss = F.cross_entropy(polarity_pred, polarity_labels)
    
    # Polarity confusion loss
    polarity_free_pred = polarity_classifier(polarity_free_recon)
    confusion_loss = -torch.mean(torch.log_softmax(polarity_free_pred, dim=1).mean(dim=1))  # Encourage classifier confusion

    return recon_loss, polarity_loss, confusion_loss

# Training loop
def train(model, data_loader, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        for batch in data_loader:
            x, polarity_labels = batch
            optimizer.zero_grad()

            # Forward pass
            embedding, polarity_recon, polarity_free_recon, polarity_pred = model(x)
            
            # Loss computation
            recon_loss, polarity_loss, confusion_loss = compute_losses(
                x, polarity_recon, polarity_free_recon, polarity_pred, polarity_labels, model.polarity_classifier
            )
            
            total_loss = recon_loss + polarity_loss + confusion_loss
            total_loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss.item()}")

# Example usage
input_dim = 300  # Example input size
hidden_dim = 128
embedding_dim = 64
output_dim = 300
num_classes = 5  # Polarity classes {-2, -1, 0, 1, 2}

model = DualDecoderModel(input_dim, hidden_dim, embedding_dim, output_dim, num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Assuming `data_loader` is a PyTorch DataLoader with batches of (input, polarity_labels)
# train(model, data_loader, optimizer)