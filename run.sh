python -m DeNSe
perl DeNSe/eval08.pl -g DeNSe/data/dev_gold -s DeNSe/data/dev_pred > result_dev.txt
perl DeNSe/eval08.pl -g DeNSe/data/test_gold -s DeNSe/data/test_pred > result_test.txt
