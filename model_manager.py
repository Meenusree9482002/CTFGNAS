import torch
from utils import data_loader
import time
import torch.nn.functional as F


class ModelManager(object):
    def __init__(self, args, log):
        self.args = args
        self.log = log
        self.args, self.data = data_loader(self.args)
        # Detect if GPU is available, otherwise use CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log.info('{}: train/val/test:{}/{}/{}, features:{}, classes:{}.'.format(
            time.strftime('%Y%m%d-%H%M%S'), sum(self.data.train_mask), sum(self.data.val_mask),
            sum(self.data.test_mask), self.args.num_node_features, self.args.num_classes))

    def train(self, model):
        data = self.data
        model = model.to(self.device)  # Move the model to the correct device (GPU or CPU)
        criterion = F.nll_loss
        optimizer = torch.optim.Adam(params=model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        
        # Set random seeds for reproducibility
        torch.manual_seed(self.args.seed)
        if self.device == torch.device('cuda'):  # Set CUDA seed only if using GPU
            torch.cuda.manual_seed(self.args.seed)
        
        val_loss, best_epoch, patience, test_acc = 99999.9999, 0, 0, 0
        try:
            for epoch in range(self.args.epochs):
                model.train()
                optimizer.zero_grad()
                # Move data to the correct device (GPU or CPU)
                out = model(data.x.to(self.device), data.edge_index.to(self.device))
                out = F.log_softmax(out, 1)
                loss = criterion(out[data.train_mask], data.y[data.train_mask].to(self.device))  # Ensure target is on correct device
                loss.backward()
                loss_train = loss.item()
                optimizer.step()

                model.eval()
                out = model(data.x.to(self.device), data.edge_index.to(self.device))
                if out.size(1) != self.args.num_classes:
                    raise
                output = F.log_softmax(out, 1)
                loss_val = criterion(output[data.val_mask], data.y[data.val_mask].to(self.device))  # Ensure target is on correct device
                loss_val = loss_val.item()

                # Move test data to the correct device and calculate accuracy
                predict = output[data.test_mask.to(self.device)].max(1)[1].type_as(data.y[data.test_mask].to(self.device))
                correct = predict.eq(data.y[data.test_mask].to(self.device)).double()
                acc_test = correct.sum() / len(data.y[data.test_mask].to(self.device))
                acc_test = acc_test.item()

                # Check if the model performance improved
                if loss_val < val_loss:
                    val_loss = loss_val
                    best_epoch = epoch + 1
                    patience = 0
                    test_acc = acc_test

                    # Read the best loss from a temp file and update it if necessary
                    with open("{}/temp.txt".format(self.args.save), 'r') as f:
                        best_loss = float(f.readlines()[0])

                    if val_loss < best_loss:
                        with open("{}/temp.txt".format(self.args.save), "w") as f:
                            f.write('{}'.format(val_loss))
                        torch.save(model, '{}/model.pth'.format(self.args.save))
                else:
                    patience += 1
                
                # Early stopping condition if the model has not improved for 150 epochs
                if patience >= 150:
                    break

            self.log.info('\t No.{:02d}: val loss:{:.4f}, test acc:{:.4f}, length:{}, best epoch:{:03d}.'.format(
                self.args.indi_no, val_loss, test_acc, model.num_layer, best_epoch))
            return round(val_loss, 4), round(test_acc, 4)
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()  # Empty GPU memory if running out of memory
                self.log.info('\t No.{}: out of memory.'.format(self.args.indi_no))
                return round(val_loss, 4), round(test_acc, 4)
            else:
                raise
