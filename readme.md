# JSON Statistics Testing and Development

May require Python 3.11 or newer. It is assumed that you are using a Python virtual environment when running the project. 

## Installation: 
To install Python 3.11 on Ubuntu, run the following commands.
```
apt install software-properties-common
add-apt-repository ppa:deadsnakes/ppa
apt install python3.11 python3.11-dev python3.11-venv
```

To install the dependencies of the project, run:
```
pip install -r requirements.txt
```


## Data:
Test data must be downloaded manually. The program expects it to be placed in a top-level directory called `data`. 
The main two datasets used during experiments were Airbnb listing data from Inside Airbnb and tweet data from the Recsys challenge repository. Some Youtube likes and dislikes data from Archive.org was used as well. 


## Running:
To run a full experiment, run the following command
```
python main.py [args]
``` 
This program accepts several command line arguments that alter its behavior. To list all supported options, run `python main.py -h`. Note that note when running in analysis mode, the argument statistics type and pruning strategies will be ignored. To run one specific configuration, you can instead modify the `test_settings` variable in `analyze.py` and set `use_test_settings` (in the same file) to `True`. 
Further and more permanent changes to the parameters can be made by modifying the settings object in `settings.py`.


To only construct a single statistics object for a single document collection, you can run
```
python create_stats.py <input_path> <output_path> [statistics type]
```
Further modifications to the parameters used by `create_stats.py` can be made by changing the settings object in `settings.py`.