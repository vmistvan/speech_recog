# a pip kezdetű sorokat alább értelemszerűen a python pip telepítőjével kell futtatni, a környezet konfigurálásakor.
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

from sklearn.model_selection import train_test_split
from flash import Trainer

from flash.audio import SpeechRecognitionData, SpeechRecognition

## az alábbi szekció biztosít hozzáférést a headerekhez és a hangfájlokhoz
## ez a datamodul beállításának szokásos módja, néha belekerül az egész egy külön osztályba.
## a datamodul tartalmai a hangfájlok a github korlátai miatt nem kerülnek bele az alkönyvtárba
## itt pl scottisch_english_female, de a magyar változat betanításához, ami elő van készítve alább
## érdemes egy magyar beszédeket tartalmazó könyvtárat létrehozni
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

## ezt a lentebbi trainer.save_checkpoint-tal együtt kell használni
## vagy betöltök egy betanított modellt, és használom, vagy, mint a jelenlegi beállításoknál
## az eredeti backbone-t töltöm be, és kimentem végül az eredményt
# model = SpeechRecognition.load_from_checkpoint(
#     "speech_recognition_model.pt"
# )

## innen akkor kell egy elemet kiválasztani, ha le akarunk tölteni egy backbone-t,
## azaz egy pretrained modellt. utána ne felejtsük elmenteni!
## tehát, ha a jelenlegi felállást használjuk, akkor a python letölti a kiválaszott backbone-t alább
## nem kell a fentebbi szekcióban modellt betölteni, viszont a tanítás végén lentebb, ne felejtsük el menteni!
# model = SpeechRecognition(backbone="facebook/wav2vec2-base-960h")
# model = SpeechRecognition(backbone="facebook/wav2vec2-large-960h-lv60")
# model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-hungarian")
model = SpeechRecognition(backbone="jonatasgrosman/wav2vec2-large-xlsr-53-hungarian")

## mivel itt pretrainelt modellekel dolgozunk, 10 korszak megfelelő, pontosabb betanulást az adatkészlet bővítésével lehet elérni.
## jelen esetben tehát több hangfájlt kell létrehozni átirattal, belerakva a fejlécbe.
trainer = Trainer(max_epochs=10, gpus=-1)

# betanítás!
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

# ez menti a finetune-d modellt.
trainer.save_checkpoint("speech_recognition_model.pt")

## adatmodul átállítása a szövegfeldolgozáshoz ha már megvolt a finetune korábban,
## akkor csak a kezdeti include sorokra és az alábbiakra van szükség!
datamodule = SpeechRecognitionData.from_files(predict_files='/content/gdrive/MyDrive/Colab Notebooks/vhm_00001_00000000011.wav', batch_size=10) # hungarian

# ez a sor hajtja végre a szövegfeldolgozást.
predictions = trainer.predict(model, datamodule=datamodule)

# kiírja, amit értett.
print(predictions)
