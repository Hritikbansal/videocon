# Downloading TEMPO-HL

[Paper](https://arxiv.org/abs/1809.01337) \
[Project Github](https://github.com/LisaAnne/TemporalLanguageRelease/tree/master)

1. The videos for the DiDeMo dataset, source dataset for TEMPO-HL, can be accessed [here](https://data.ciirc.cvut.cz/public/projects/LisaAnne/DiDeMoRelease/).
2. The captions are present [here](https://data.ciirc.cvut.cz/public/projects/LisaAnne/TEMPO/).
3. We process the TEMPO captions to filter out `TEMPO-Template` and retain `TEMPO-HL`. 
4. We process the original DiDeMo videos to merge two 5 second segments. There is one caption corresponding to one processed video. This is in accordance with the processing steps in the original github. The code for the same is present in [create_tempo_dataset.py](src/create_tempo_dataset.py) 
5. We filter out all the videos with extensions other than `.mp4, .avi, .mov`

In case you are facing difficulties with the `TEMPO-HL` data, please drop me an [email](hbansal@g.ucla.edu) and I will be happy to share the processed data. 
