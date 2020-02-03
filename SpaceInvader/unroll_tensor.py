import torch 

def pytorch_rolling_window(x, window_size):
    # unfold dimension to make our rolling window
    return x.unfold(0,window_size,1)

x = torch.range(1,20).view(4,5)
z = pytorch_rolling_window(x,3)

print("Final Shape: ", z.shape)
print(z)

n = z.permute(0, 2, 1)
print("New Shape: ", n.shape)
print(n)