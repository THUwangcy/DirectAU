# DirectAU

Source code of the paper "Towards Representation Alignment and Uniformity in Collaborative Filtering" in KDD'22.

This method is easy to implement as follows (PyTorch-style):

```python
@staticmethod
def alignment(x, y, alpha=2):
    x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

@staticmethod
def uniformity(x, t=2):
    x = F.normalize(x, dim=-1)
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

def calculate_loss(self, user, item):
    user_e, item_e = self.encoder(user, item)  # [bsz, dim]
    align = self.alignment(user_e, item_e)
    uniform = (self.uniformity(user_e) + self.uniformity(item_e)) / 2
    loss = align + self.gamma * uniform
    return loss
```

A runnable project with built-in datasets and example commands will come soon.



## Contact

Chenyang Wang ([THUwangcy@gmail.com](mailto:THUwangcy@gmail.com))
