# Neural-based Dependency Parser ([Dependency Parsing as Head Selection](http://aclweb.org/anthology/E17-1063), Zhang et al., EACL 2017)
## Requirements
### Framework
 - python (<= 3.6)
 - pytorch (<= 0.3.0)
 
### Packages
 - torchtext
 - toml
 
 you can install these packages by `pip instlall -r requirements.txt`.
## Performance

| PBT English | Reported score | Our implementation |
|:---:|:---:|:---:|
| DEV | 94.17 | 94.18 |
| TEST | 94.02 | 94.13 |

The training time is approximately 9 minutes for 5 iterations with the batch size equal to 16.
