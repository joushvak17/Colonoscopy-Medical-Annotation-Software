<div align="center">
  <h1>Colonoscopy Medical Annotation Software 💉</h1>
  <p align="center"><strong>Medical annotation software for colonoscopy images that utilizes C# and deep learning</strong></p>
</div>

## 📑 File Outline 

### [Colonoscopy Medical Annotation Project](https://github.com/joushvak17/Colonoscopy-Medical-Annotation-Software/tree/master/Colonoscopy%20Medical%20Annotation%20Project)
- Models.ipynb: Notebook that creates Python scipts for the models
- Run.ipynb: Notebook that creates Python scripts for the deep learning workflow

### [Colonoscopy Medical Annotation Project/modular](https://github.com/joushvak17/Colonoscopy-Medical-Annotation-Software/tree/master/Colonoscopy%20Medical%20Annotation%20Project/modular)
- data_setup.py: Defines the functionality for creating PyTorch dataloaders for the multi-class classification dataset
- engine.py: Defines functions for training and testing PyTorch models
- utils.py: Defines functions that contain various utility functions for PyTorch model training and saving 
- train.py: Defines the training scipt for the PyTorch model
- mlflow_train.py: Defines the training script for the PyTorch model with MLflow

### [Colonoscopy Medical Annotation Project/modular/models](https://github.com/joushvak17/Colonoscopy-Medical-Annotation-Software/tree/master/Colonoscopy%20Medical%20Annotation%20Project/modular/models)
- baseline_model.py: Baseline model Python script
- TinyVGG_model.py: Tiny VGG model Python script
- vgg19_model.py: VGG19 model Python script

## 📜 License

The Colonoscopy Medical Annotation Software is under the MIT License. See the [License](License) file for details.