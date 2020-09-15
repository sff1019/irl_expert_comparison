# エキスパートデータに関する実験

## 概要
GAILで用いるエキスパートデータに関する実験を行った。

## 環境構築
```
$ pip install -r requirements.txt
$ python setup.py install  # install torchrl
```

## 実行例
```
# collect expert data
$ python scripts/trpo_pendulum.py

# train gail
$ python scripts/gail_trpo_pendulum.py 
```

## 参考
- 強化学習フレームワーク: [garage](https://github.com/rlworkgroup/garage)
- GAILの参考：[inverse_rl](https://github.com/justinjfu/inverse_rl)
