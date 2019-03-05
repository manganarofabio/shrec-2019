How to install
=====================
In "Shrec Competition" directory you can find "requirements.txt" containing all the modules you need to run the python(3.6.4) script (Eval.py).
To install requirements open a terminal in the current directory and type:
	python -m venv "virtual_env_name" (optional if you don't want to create a virtual enviroment)
	"virtual_env_name"\Scripts\activate
	pip install -r requirements-{windows/linux}.txt
	
*requirements-windows for windows os; requirements-linux for linux os.

How to use
=====================
In "Shrec Competition" directory open a terminal and type the following command:
	python eval.py	--dataset_path	"Test" --out_file "results.txt"
	
*--dataset_path: path to directory containing test files. The directory has to follow this structure:
	./Test
		--file1.txt
		--file2.txt
		...
		--fileN.txt

*--out_file: name of results file(it will be saved in the same directory).

Output
======================
the script produces 2 different output files:
	
1) results.txt: results file(csv) formatted as
	trajectory_number;predicted_gesture_label;predicted_gesture_start;predicted_gesture_end

2) results_timestamp.txt: resultsfile formatted using timestamp as frame temporal id, as
	trajectory_number;predicted_gesture_label;predicted_gesture_start_timestamp;predicted_gesture_end_timestamp
			
if the script does not predict a gestur_label for a given trajectory_number, it marks the trajectory_number as:
trajectory_number;-1
			


