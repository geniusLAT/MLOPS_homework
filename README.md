To run the code:
0\. Execute the git clone command for this project ( https://github.com/geniusLAT/MLOPS_homework.git) and navigate to the project directory cd MLOPS_homework
1\. Install dependencies from the requirements\.txt file located in the project root \- pip install \-r requirements\.txt
2\. Navigate to the lab1 directory \- cd lab1
3\. Look into the pipeline\.sh script
4\. Then run the pipeline\.sh script with the command ``./pipeline.sh`` after ensuring that the file is executable \(ls \-la && chmod u\+x pipeline\.sh\)

To run individual scripts from the initial directory before step 0:

python3 data\_creation\.py \- generates data and writes to corresponding files \(training and testing/validation\)

python3 model\_preprocessing\.py \- prepares data for further processing

python3 model\_preparation\.py \- creates and trains a model on the training data, then writes it to a separate file

python3 model\_testing\.py \- tests the model's metrics on the testing data