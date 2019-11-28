import torch
import settings
import fastai.vision
assert torch.cuda.is_available()

# print(dir(fastai.vision))

def run():
    data = fastai.vision.ImageDataBunch.from_folder(
        settings.BASE_PATH, train='training', valid='validation',
        size=(169, 300)
    )

    learn = fastai.vision.cnn_learner(data, fastai.vision.models.resnet18, metrics=fastai.vision.accuracy)
    learn.fit(1)

    interpretation = fastai.vision.ClassificationInterpretation.from_learner(learn)
    interpretation.plot_top_losses(9, figsize=(6, 6))
    interpretation.plot_confusion_matrix()

    img = learn.data.train_ds[0][0]
    learn.predict(img)

if __name__ == '__main__':
    run()
