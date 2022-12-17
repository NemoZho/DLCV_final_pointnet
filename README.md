# DLCV Final

## Dataset
將dataset資料夾與本專案資料夾放在同一個路徑下

## How to Run

```
$ python3 DLCV_final_pointnet/utils/train_segmentation.py
```

## Description

- dataset.py會export一個ScanNet200Dataset，每次output出point_set, cls兩個tensor，point_set形狀為<點數量x6>，其中6為xyzrgb的組合（rgb有經過標準化），cls形狀為<點數量>，為每個點的標籤。
- 標籤使用201個class，分別為0~200，其中1到200會被map進官方的標籤裡，0代表不確定或並沒有在200個標籤裡。
- batch_size只能為1，因為每個scene的點數量差距過大，padding會導致gpu記憶體爆炸。
- 將train_set切分為train和valid，ratio可以在train_segmentation.py裏面定義。
- 最後階段計算miou分數疑似會有bug，待修正。
