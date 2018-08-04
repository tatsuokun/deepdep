# Neural-based Dependency Parser ([Dependency Parsing as Head Selection](http://aclweb.org/anthology/E17-1063), Zhang et al., EACL 2017)

This is a PyTorch implementation of the neural-based dependency parser as in [Dependency Parsing as Head Selection](http://aclweb.org/anthology/E17-1063) achieved nearly state-of-the-art on dependency parsing in early 2017.

## Requirements
### Framework
 - python (<= 3.6)
 - pytorch (<= 0.4.0)
 - perl (<= 5.0) it's used only for evaluation, not training phase
 
### Packages
 - torchtext
 - toml
 - allennlp
 
 You can install these packages by `pip install -r requirements.txt`.
 
### Dataset
Put conllx format dataset (for example PTB English as in the original paper) in `deepdep/data`.

If you want to run this program quickly, please make your directory structure as below.

Otherwise, edit `config.toml` so you can run the program with your dataset.

 ```
deepdep
│
├ data
│　└ ptb.conllx
│　   ├ train.conllx.txt
│　   ├ dev.conllx.txt
│　   └ test.conllx.txt
│
├ DeNSe
│
```

## How to run

```
python -m DeNSe --config config.toml --gpu-id 0
perl DeNSe/eval08.pl -g results/dev_gold -s results/dev_pred > result_dev.txt
perl DeNSe/eval08.pl -g results/test_gold -s results/test_pred > result_test.txt
```

The trained model is saved in `deepdep/models`.

## Performance

| PBT English | Reported score | Our implementation | Out implementation + ELMo |
|:---:|:---:|:---:|:---:|
| DEV | 94.17 | 94.18 | 94.90 |
| TEST | 94.02 | 94.13 | 94.95 |

The estimated training time is approximately 9 minutes for 5 iterations with the batch size equal to 16. (I used TITAN X)
## Reference

```
@InProceedings{E17-1063,
  author = 	"Zhang, Xingxing
		and Cheng, Jianpeng
		and Lapata, Mirella",
  title = 	"Dependency Parsing as Head Selection",
  booktitle = 	"Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 1, Long Papers",
  year = 	"2017",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"665--676",
  location = 	"Valencia, Spain",
  url = 	"http://aclweb.org/anthology/E17-1063"
}
```
```
@InProceedings{N18-1202,
  author = 	"Peters, Matthew
		and Neumann, Mark
		and Iyyer, Mohit
		and Gardner, Matt
		and Clark, Christopher
		and Lee, Kenton
		and Zettlemoyer, Luke",
  title = 	"Deep Contextualized Word Representations",
  booktitle = 	"Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)",
  year = 	"2018",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"2227--2237",
  location = 	"New Orleans, Louisiana",
  url = 	"http://aclweb.org/anthology/N18-1202"
}

```
