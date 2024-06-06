# pip install pytorch-lightning
# pip install lightning-flash
# pip install 'lightning-flash[audio,image, video, text]'

# pip install huggingsound

# pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

import pandas as pd
import random
import torch
import flash

# from huggingsound import SpeechRecognitionModel

# model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-hungarian")
from sklearn.model_selection import train_test_split
from flash import Trainer

from flash.audio import SpeechRecognitionData, SpeechRecognition


random.seed(10)
path_txt = "/content/gdrive/MyDrive/Colab Notebooks/"

df_hungarian = pd.read_csv(path_txt+'line_index.csv', header=None, delimiter="/", names=['not_required', 'speech_files', 'targets'])
df_hungarian = df_hungarian.sample(frac=0.8)
print(df_hungarian.shape)
df_hungarian.head()

df_hungarian = df_hungarian[['speech_files', 'targets']]
df_hungarian['speech_files'] = df_hungarian['speech_files'].str.lstrip()
df_hungarian['speech_files'] = path_txt+df_hungarian['speech_files'].astype(str) + '.wav'
df_hungarian.head()
random.seed(10)
train_hungarian, test_hungarian_raw = train_test_split(df_hungarian, test_size=0.2)
test_hungarian = test_hungarian_raw['speech_files']
test_hungarian.head()
train_hungarian.to_csv('train_hungarian.csv')
test_hungarian.to_csv('test_hungarian.csv')

datamodule = SpeechRecognitionData.from_csv("speech_files","targets",train_file="train_hungarian.csv",predict_file="test_hungarian.csv",batch_size=10)

print ("ez van:",SpeechRecognition.available_backbones())

# model = SpeechRecognition.load_from_checkpoint(
#     "speech_recognition_model.pt"
# )

# model = SpeechRecognition(backbone="facebook/wav2vec2-base-960h")
# model = SpeechRecognition(backbone="facebook/wav2vec2-large-960h-lv60")
# model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-hungarian")
model = SpeechRecognition(backbone="jonatasgrosman/wav2vec2-large-xlsr-53-hungarian")

trainer = Trainer(max_epochs=10, gpus=-1)

trainer.finetune(model, datamodule=datamodule, strategy="freeze")

trainer.save_checkpoint("speech_recognition_model.pt")

datamodule = SpeechRecognitionData.from_files(predict_files='/content/gdrive/MyDrive/Colab Notebooks/vhm_00001_00000000011.wav', batch_size=10) # hungarian

predictions = trainer.predict(model, datamodule=datamodule)
print(predictions)