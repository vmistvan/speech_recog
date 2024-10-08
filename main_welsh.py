import os
import pandas as pd
import random
import sklearn
import torch
import flash

from sklearn.model_selection import train_test_split
from flash import Trainer
# from lightning.pytorch import Trainer
from flash.audio import SpeechRecognitionData, SpeechRecognition
## print("Helló, mi?", pl.LIGHTNING_LOGO, pl.__version__)

# harc a memóriáért. legyen jóó videókártyád, 16G, és hali
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048"
#
# print("pandas version:",pd.__version__)
# print("torch version:",torch.__version__)
# print("flash version:",flash.__version__)
# print("cuda version:",torch.cuda_version)
# print("torch / cuda version:",torch.version.cuda)
# print("cuda available:",torch.cuda.is_available())

random.seed(10)
path_txt = "./welsh_english_female/"
df_welsh = pd.read_csv(path_txt+"line_index.csv", header=None, names=['not_required', 'speech_files', 'targets'])
df_welsh = df_welsh.sample(frac=0.08)
print(df_welsh.shape)
df_welsh.head()

df_welsh = df_welsh[['speech_files', 'targets']]
df_welsh['speech_files'] = df_welsh['speech_files'].str.lstrip()
df_welsh['speech_files'] = path_txt+df_welsh['speech_files'].astype(str) + '.wav'
print('a elsők:', df_welsh.head())
random.seed(10)
train_welsh, test_welsh_raw = train_test_split(df_welsh, test_size=0.2)
test_welsh = test_welsh_raw['speech_files']
test_welsh.head()
train_welsh.to_csv('train_welsh.csv')
test_welsh.to_csv('test_welsh.csv')

datamodule = SpeechRecognitionData.from_csv("speech_files","targets",train_file="train_welsh.csv",predict_file="test_welsh.csv",batch_size=10)

print ("ez van:",SpeechRecognition.available_backbones())

# három lehetőség van: betölteni, ha van, a nagy és a normál modell. kezdjük a normállal!
# model = SpeechRecognition.load_from_checkpoint(
#      "speech_recognition_model.pt"
# )
# model = SpeechRecognition(backbone="facebook/wav2vec2-large-960h-lv60")
model = SpeechRecognition(backbone="facebook/wav2vec2-base-960h")

# nem, már nem kell, sőt nem is szabad a jó eredményhez precision paramétert megadni!
# trainer = Trainer(max_epochs=20, gpus=-1, precision=16)
trainer = Trainer(max_epochs=10, gpus=-1)

# trainer.finetune(model, datamodule=datamodule, strategy="no_freeze")
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

# ha még nincs, akkor el lehet menteni a testreszabott modellt. utána nem is kell betanulni, mondhatni
# az összes fenti sor feleslegessé válik!
trainer.save_checkpoint("speech_recognition_model.pt")

# # trainer.predict(model, datamodule=datamodule)
# print(datamodule.train_dataloader)
# predictions = trainer.predict(model, datamodule=datamodule, output=str)
# print(predictions)

datamodule = SpeechRecognitionData.from_files(predict_files='wef_05223_01497543874.wav', batch_size=10) # welsh

predictions = trainer.predict(model, datamodule=datamodule)
print(predictions)
