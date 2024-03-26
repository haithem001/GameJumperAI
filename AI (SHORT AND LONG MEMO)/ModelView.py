import torch
model = torch.load('./model/model.pth')
model2 = torch.load('./model/SAVE.pth')
print (model2)
print("-------------------------------------")
print(model)