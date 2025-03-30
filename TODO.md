
## dataset & dataloading problem
- [ ] Enable async GPU data processing and training
- [ ] Load checkpoints before training
- [x] Verify by visualizing the image after pre-processing
- [ ] enlarge the dataset with more data source
- [ ] set expriments for training on 2D
- [x] Improve the loading efficiency: speed up with monai
- [x] Apply pre-trained model from [Med3D models](https://github.com/Tencent/MedicalNet)
- [x] Data agumentation in pre-processing pipeline with [torchio](https://github.com/TorchIO-project/torchio)
    - [x] random rotation within 60 degrees
    - [x] random shear of scale 0.1
    - [x] random horizontal and vertical flips
    - [x] random intensity scaling in range 0.8-1.2
- [x] resolve the length of z-axis image cross the sequence mismatch problem
    - [x] Detect the distribution of the tumor location
