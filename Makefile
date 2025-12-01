install:
	conda config --append channels conda-forge
	conda env update -n snake-ql --file environment.yml --prune