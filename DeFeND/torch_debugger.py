import torch


class ModelDebugger(torch.nn.Module):
    def __init__(self, model, device):
        super(ModelDebugger, self).__init__()
        self.model = model
        self.device = device
        self.model = self.model.to(self.device)
        
    def forward(self, x):
        x = x.to(self.device)
        outputs = self.model(x)
        # Perform checks on the output
        # For example, check if the output has any NaN values
        for output in outputs:
            if torch.isnan(output).any():
                print("Output has NaN values!")
        # Return the output
        return output
    
    def check_gradients(self):
        # Perform checks on the gradients
        # For example, check if any gradients have exploded
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"Gradients for parameter {name} have exploded!")
    
    def check_weights(self):
        # Perform checks on the weights
        # For example, check if any weights have become too large
        for name, param in self.model.named_parameters():
            if torch.norm(param) > 1000:
                print(f"Weight for parameter {name} has become too large!")
    

    def check_trainable_params(self):
        # Check if the model has any trainable parameters
        if len(list(self.model.parameters())) == 0:
            print("Model has no trainable parameters!")
    
    def print_trainable_weights(self):
        # print the trainable weights
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)
    
    def apply_checks(self):
        # Apply all the checks
        self.check_gradients()
        self.check_weights()
        self.check_trainable_params()
        self.print_trainable_weights()
    

