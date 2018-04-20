# Neural-based Dependency Parser ([Dependency Parsing as Head Selection](http://aclweb.org/anthology/E17-1063), Zhang et al., EACL 2017)

This is a PyTorch implementation of the neural-based dependency parser as in [Dependency Parsing as Head Selection](http://aclweb.org/anthology/E17-1063) achieved nearly state-of-the-art on dependency parsing in early 2017.

## Requirements
### Framework
 - python (<= 3.6)
 - pytorch (<= 0.3.0)
 - perl (<= 5.0) it's used only for evaluation, not training phase
 
### Packages
 - torchtext
 - toml
 
 You can install these packages by `pip instlall -r requirements.txt`.
 
### Dataset
Put conllx format dataset (for example PTB English as in the original paper) in `DeNSe/data/`

If you want to run this program quickly, please make your directory structure as below.
 ```
DeNSe
│
│
├ data
│　└ ptb.conllx
│　   ├ train.conllx.txt
│　   ├ dev.conllx.txt
│　   └ test.conllx.txt
│
```

## How to run

```
python -m DeNSe --config config.toml --gpu-id 0
perl DeNSe/eval08.pl -g DeNSe/data/dev_gold -s DeNSe/data/dev_pred > result_dev.txt
perl DeNSe/eval08.pl -g DeNSe/data/test_gold -s DeNSe/data/test_pred > result_test.txt
```

The trained model are saved in `DeNSe/models`

## Performance

| PBT English | Reported score | Our implementation |
|:---:|:---:|:---:|
| DEV | 94.17 | 94.18 |
| TEST | 94.02 | 94.13 |

The training time is approximately 9 minutes for 5 iterations with the batch size equal to 16.

## Reference

```
@InProceedings{zhang-cheng-lapata:2017:EACLlong,
  author    = {Zhang, Xingxing  and  Cheng, Jianpeng  and  Lapata, Mirella},
  title     = {Dependency Parsing as Head Selection},
  booktitle = {Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 1, Long Papers},
  month     = {April},
  year      = {2017},
  address   = {Valencia, Spain},
  publisher = {Association for Computational Linguistics},
  pages     = {665--676},
  url       = {http://www.aclweb.org/anthology/E17-1063}
}
```
