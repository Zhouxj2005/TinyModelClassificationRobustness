# mobilenet_v2_standard（200epoch）
Corruption: gaussian_noise       | Acc: 4.30%
Corruption: shot_noise           | Acc: 4.05%
Corruption: impulse_noise        | Acc: 6.62%
Corruption: defocus_blur         | Acc: 5.07%
Corruption: glass_blur           | Acc: 8.11%
Corruption: motion_blur          | Acc: 13.25%
Corruption: zoom_blur            | Acc: 41.84%
Corruption: snow                 | Acc: 35.13%
Corruption: frost                | Acc: 34.27%
Corruption: fog                  | Acc: 54.39%
Corruption: brightness           | Acc: 64.01%
Corruption: contrast             | Acc: 41.68%
Corruption: elastic_transform    | Acc: 53.22%
Corruption: pixelate             | Acc: 48.32%
Corruption: jpeg_compression     | Acc: 47.59%

----------------------------------------
Checkpoint Clean Acc: 69.35%
Robust Accuracy (mAcc): 30.79%

----------------------------------------


# mobilenet_v2_augmix（200epoch）

Corruption: gaussian_noise       | Acc: 15.71%
Corruption: shot_noise           | Acc: 15.77%
Corruption: impulse_noise        | Acc: 29.12%
Corruption: defocus_blur         | Acc: 11.69%
Corruption: glass_blur           | Acc: 21.17%
Corruption: motion_blur          | Acc: 24.41%
Corruption: zoom_blur            | Acc: 61.87%
Corruption: snow                 | Acc: 41.73%
Corruption: frost                | Acc: 39.30%
Corruption: fog                  | Acc: 57.03%
Corruption: brightness           | Acc: 66.70%
Corruption: contrast             | Acc: 54.62%
Corruption: elastic_transform    | Acc: 62.59%
Corruption: pixelate             | Acc: 60.28%
Corruption: jpeg_compression     | Acc: 59.21%

----------------------------------------
Checkpoint Clean Acc: 70.60%
Robust Accuracy (mAcc): 41.41%

----------------------------------------
# mobilenet_v2_distill(200epoch)-$\alpha=15,lr=0.05$

Corruption: gaussian_noise       | Acc: 19.03%
Corruption: shot_noise           | Acc: 18.09%
Corruption: impulse_noise        | Acc: 29.47%
Corruption: defocus_blur         | Acc: 9.33%
Corruption: glass_blur           | Acc: 18.07%
Corruption: motion_blur          | Acc: 23.10%
Corruption: zoom_blur            | Acc: 64.14%
Corruption: snow                 | Acc: 45.32%
Corruption: frost                | Acc: 45.03%
Corruption: fog                  | Acc: 62.10%
Corruption: brightness           | Acc: 70.95%
Corruption: contrast             | Acc: 55.99%
Corruption: elastic_transform    | Acc: 65.02%
Corruption: pixelate             | Acc: 60.87%
Corruption: jpeg_compression     | Acc: 61.15%

----------------------------------------
Checkpoint Clean Acc: 74.20%
Robust Accuracy (mAcc): 43.18%

----------------------------------------

# mobilenet_v2_distill_"improved"(200epoch)-$\alpha=20,\beta=500,lr=0.05$

Corruption: gaussian_noise       | Acc: 13.79%
Corruption: shot_noise           | Acc: 13.94%
Corruption: impulse_noise        | Acc: 24.23%
Corruption: defocus_blur         | Acc: 11.06%
Corruption: glass_blur           | Acc: 20.73%
Corruption: motion_blur          | Acc: 22.89%
Corruption: zoom_blur            | Acc: 59.46%
Corruption: snow                 | Acc: 40.02%
Corruption: frost                | Acc: 37.34%
Corruption: fog                  | Acc: 55.33%
Corruption: brightness           | Acc: 64.14%
Corruption: contrast             | Acc: 51.73%
Corruption: elastic_transform    | Acc: 60.72%
Corruption: pixelate             | Acc: 59.11%
Corruption: jpeg_compression     | Acc: 57.95%

----------------------------------------
Checkpoint Clean Acc: 68.56%
Robust Accuracy (mAcc): 39.50%

----------------------------------------

# resnet18_augmix（200epoch）

Corruption: gaussian_noise       | Acc: 17.25%
Corruption: shot_noise           | Acc: 16.76%
Corruption: impulse_noise        | Acc: 33.00%
Corruption: defocus_blur         | Acc: 12.23%
Corruption: glass_blur           | Acc: 21.69%
Corruption: motion_blur          | Acc: 30.20%
Corruption: zoom_blur            | Acc: 70.45%
Corruption: snow                 | Acc: 47.49%
Corruption: frost                | Acc: 48.93%
Corruption: fog                  | Acc: 66.11%
Corruption: brightness           | Acc: 73.82%
Corruption: contrast             | Acc: 66.18%
Corruption: elastic_transform    | Acc: 68.86%
Corruption: pixelate             | Acc: 64.14%
Corruption: jpeg_compression     | Acc: 63.05%

----------------------------------------

Checkpoint Clean Acc: 76.30%
Robust Accuracy (mAcc): 46.68%

----------------------------------------



# resnet18-Prime

Corruption: gaussian_noise       | Acc: 51.43%
Corruption: shot_noise           | Acc: 48.44%
Corruption: impulse_noise        | Acc: 35.76%
Corruption: defocus_blur         | Acc: 13.72%
Corruption: glass_blur           | Acc: 23.90%
Corruption: motion_blur          | Acc: 29.98%
Corruption: zoom_blur            | Acc: 69.54%
Corruption: snow                 | Acc: 51.05%
Corruption: frost                | Acc: 58.19%
Corruption: fog                  | Acc: 67.43%
Corruption: brightness           | Acc: 75.15%
Corruption: contrast             | Acc: 71.21%
Corruption: elastic_transform    | Acc: 69.26%
Corruption: pixelate             | Acc: 66.78%
Corruption: jpeg_compression     | Acc: 66.17%

Checkpoint Clean Acc: 77.60% 
Robust Accuracy (mAcc): 53.20%