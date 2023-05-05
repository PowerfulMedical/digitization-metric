## Expected folder structure

Folder with provided data:

```
base_dir/
  data/
    leads.npz
    rhythms.npz
  visual_data/
    augmentation_JPEG_compression_11_percent/
      img_1_page_0.jpeg
      ...
    ...
  metadata.csv
```

Folder with digitized data (same structure as visual data, but for each image provide one json file, even for multipage ECGs you should provide just one file).:

```
digi_ecg_dir/
  augmentation_JPEG_compression_11_percent/
    img_1.json
    img_2.json
  ...
```

If you cannot digitize some ecg, you might omit the corresponding file (you will get score 0 for that ecg).


JSON format is as follows:
```
{
  "I": {"ecg": [<list of values in mV>], "fs": <sampling frequency>},
  "II": {...},
  ...
  "V6": {...}
}
```

Rhythm leads (if present) are marked as "rhythm1", "rhythm2", "rhythm3".

Example of such output is in `example_digitization` folder.

## Running evaluation

`python calc.py <dataset_folder> <digitized_folder> <output_csv>`

`python summarize.py <output_csv>`

