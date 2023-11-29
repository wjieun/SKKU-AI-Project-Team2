## Segmentation

### Dataset

Dataset으로는 PLSU 데이터를 사용하였습니다.

Dataset 전처리 과정은 아래와 같습니다.

1.PLSU Datasset: 1039개 (image, mask)
2.data clining작업 => mask중 선X, 선2개인 mask 삭제 : 총 1016개
3.data 분리 => train:val:test를 406:305:305 비율로 분리
4.data augmentation작업 => train:val:test를 1016:305:305 비율로 분리

최종 data 비율: 72.69:13.65:13.65
