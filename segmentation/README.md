1.PLSU Datasset: 1039개 (image, mask)
2.Data clining작업(마스크 없는것 삭제): data clining.py + 수작업(20개)
3.Data preprocessing작업 (rotate, flip): 총4064개 
4.각 폴더는 실험한 결과.
4.최종 선택 모델: ViT + Unet, Loss: Jaccard Loss, Scheduler: CosineLR
