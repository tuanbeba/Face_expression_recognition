# Face Expression
This repository using [Fer2013](https://www.kaggle.com/datasets/deadskull7/fer2013) dataset and pytorch framework 


The key components of project:

- Model use a pretrained efficient_b0 for feature extraction
- Data preprocessing: converting .csv files to .png format and splitting the data into training and validation sets, which are saved in the data folder.
- Model training and evaluation.
- Saving model weights and visualizing the results after each training session.

## Installation

<div style="position: relative; display: inline-block;">
    <button onclick="copyCode()" style="position: absolute; right: 10px; top: 10px;">📋</button>
    <pre>
        <code>
conda env create -f environment.yml
conda env list
        </code>
    </pre>
</div>

<script>
function copyCode() {
    const code = `conda env create -f environment.yml\nconda env list`;
    navigator.clipboard.writeText(code).then(() => {
        alert('Code copied to clipboard!');
    });
}
</script>
## Training

You have to download the fer2013.csv file and put it in the data folder then run the train.py file

## Result
[Accuracy](.checkpoints\1\accuracy_curve.png)

[Loss](.checkpoints\1\loss_curve.png)


