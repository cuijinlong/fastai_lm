import torch
import lightning as L

class ToyExample(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch):
        loss = self.model(batch).sum()
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters())

if __name__ == "__main__":
    model = torch.nn.Linear(32, 2)
    pl_module = ToyExample(model)
    train_dataloader = torch.utils.data.DataLoader(torch.randn(8, 32))

    trainer = L.Trainer()
    trainer.fit(pl_module, train_dataloader)