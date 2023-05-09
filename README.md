# ECG Evaluation Tool
This evaluation tool is designed as a supplemental material to the paper titled 
"_Development of a Diverse ECG Dataset for Testing and Evaluating Digitization Solutions_" by Viera Kresnakova MSc, PhD, 
Andrej Iring MSc, Vladimir Boza MSc, PhD, Boris Vavrik MSc, Michal Hojcka MSc, PhD, Simon Rovder MInf, Timotej Palus MSc,
Robert Herman MD, Martin Herman, Felix Bauer BSc, and Viktor Jurasek. 

The tool evaluates the performance of digitized ECG data by comparing it to a 
gold standard ECG dataset. It calculates the normalized pixel distance scores for each 
ECG and provides summary statistics to support the findings presented in the paper.

## Objective
The objective of this evaluation tool is to provide a quantitative comparison between the 
digitized ECGs and the gold standard ECG dataset. This comparison allows the readers to 
understand the effectiveness of the digitization methods discussed in the paper and to potentially reproduce the results
or apply the methods to their own datasets.

## Methodology
The evaluation tool calculates the normalized pixel distance scores for each ECG by comparing the digitized ECG data to
the gold standard dataset. The comparison is performed using the `compare_ecgs` function, which calculates error scores 
between the two ECGs. The normalized score is then calculated as `(1 - digi_res / flat_res)`, where `digi_res` is 
the error score for the digitized ECG, and `flat_res` is the error score for a flat (zero-valued) ECG. 
Lower normalized scores indicate better performance in terms of digitization accuracy.

## Results Interpretation
The summary statistics provided by the evaluation tool include the average normalized pixel distance scores for 
each dataset and the total average score. These results can be used to compare different digitization methods, 
assess the impact of various preprocessing techniques, or support the conclusions drawn in the paper.

_Note_: This evaluation tool is intended to be used as a supplement to the paper and should be used in conjunction 
with the paper's main content to fully understand the context and implications of the results generated by the tool.

## Expected folder structure

### Provided Data Folder:

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
### Digitized Data Folder:

The folder structure should be the same as the visual data folder, but with JSON files for each image. 
For multipage ECGs, provide just one file.

```
digi_ecg_dir/
  augmentation_JPEG_compression_11_percent/
    img_1.json
    img_2.json
  ...
```

If you cannot digitize an ECG, you may omit the corresponding file (you will get a score of 0 for that ECG).

### JSON Format:

The JSON format should be as follows:

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
Calculate the normalized pixel distance scores for each ECG:

`python calc.py <dataset_folder> <digitized_folder> <output_csv>`

Summarize the scores and print the results:

`python summarize.py <output_csv>`

This will generate a summary of the normalized pixel distance scores by dataset and the total average score.




