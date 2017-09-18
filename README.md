# officeActionClassification

## 概要
- trial1: rnnやcnnなどを自分で学習させるトライ。あまり良い結果は出ていない
- trial2: deepPoseを使用して精度を上げるトライ。評価サンプルで40%程度の精度

## 学習データ
- MSR Daily Activityをダウンロード: http://users.eecs.northwestern.edu/~jwa368/my\_data.html
- analysis.pyに対して、../../data/a01\_s07\_e02\_rgb.aviという形に展開

## trial2実行前準備
- analysis.pyに対して、model/model.h5となるよう、以下からdeepPoseのモデルをダウンロード: https://github.com/michalfaber/keras\_Realtime\_Multi-Person\_Pose\_Estimation
