## training pipeline improving
- [ ] More felxible way for train, val and test indicies (current may too long)
- [ ] Cross validation runner that auto create folds by combining train and val
- [ ] More felxible way for clinical feature columns indexing (some columns may too long)
- [x] Load checkpoints before training

## model design improving
- [ ] Expriments for training on [2D MRI](https://europepmc.org/article/MED/37370560)
- [ ] Expriments for [tabular to image](https://github.com/zhuyitan/IGTD?tab=readme-ov-file)
- [ ] Feature selecting for clinical features
- [ ] Baseline cross validation with clinical data and MRI with Cross attention
- [x] Apply pre-trained model from [Med3D models](https://github.com/Tencent/MedicalNet)

## dataset & dataloading problem
- [ ] Missing clinical data imputing in dataloading pipeline (simple way)
- [ ] Enlarge the dataset with more data source
- [x] Enable async GPU data processing and training
- [x] clinical data only experiments
- [x] Verify by visualizing the image after pre-processing
- [x] Improve the loading efficiency: speed up with monai
- [x] Data agumentation in pre-processing pipeline with [torchio](https://github.com/TorchIO-project/torchio)
    - [x] random rotation within 60 degrees
    - [x] random shear of scale 0.1
    - [x] random horizontal and vertical flips
    - [x] random intensity scaling in range 0.8-1.2
- [x] resolve the length of z-axis image cross the sequence mismatch problem
    - [x] Detect the distribution of the tumor location
