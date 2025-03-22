
## dataset & dataloading problem
- [ ] Data agumentation in pre-processing pipeline with [torchio](https://github.com/TorchIO-project/torchio)
    - [ ] random rotation within 60 degrees
    - [ ] random shear of scale 0.1
    - [ ] random horizontal and vertical flips
    - [ ] random intensity scaling in range 0.8-1.2
- [ ] Load checkpoints before training
- [ ] Verify by visualizing the image after pre-processing
- [ ] enlarge the dataset with more data source
- [ ] set expriments for training on 2D
- [x] resolve the length of z-axis image cross the sequence mismatch problem
    - [x] Detect the distribution of the tumor location
